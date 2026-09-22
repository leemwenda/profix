"""Signal generation (pure functions, no execution concerns).

TIMING CONTRACT (identical in the MQL5 EA and Pine script):
  * A signal is evaluated at the CLOSE of execution-timeframe bar `i`.
    It may use only data of bars <= i.
  * HTF trend uses only HTF bars whose CLOSE time <= (open time of bar i) + 1 hour,
    i.e. bars that are fully closed at the moment the decision is taken.
  * The order is executed at the OPEN of bar i+1 (only if bar i+1 opens exactly one
    hour later - no weekend/holiday gap - checked by the engine).
  * Nothing is repainted: values at row i never change when later rows are appended
    (verified by tests/test_no_lookahead.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import StrategyParams
from .data import HOUR_NS, resample_htf
from .specs import SymbolSpec


def to_ns(idx: pd.DatetimeIndex) -> np.ndarray:
    """Robust int64 nanoseconds-since-epoch (pandas>=3 may store other resolutions)."""
    return idx.tz_convert("UTC").tz_localize(None).to_numpy().astype("datetime64[ns]").astype(np.int64)


@dataclass(frozen=True)
class Indicators:
    t: np.ndarray        # bar open time, ns
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray      # SMA of true range (== MT5 iATR)
    hh: np.ndarray       # highest high of the N bars BEFORE the signal bar
    ll: np.ndarray
    clv: np.ndarray      # close location value in [0,1]
    rng: np.ndarray      # bar range
    trend: np.ndarray    # +1 / -1 / 0 from closed HTF bars
    regime: np.ndarray   # ATR percentile tercile (0,1,2), -1 unknown (analysis only)
    hour_next: np.ndarray  # UTC hour of the (assumed) entry bar
    wd_next: np.ndarray    # weekday of entry bar (Mon=0)


def indicator_key(p: StrategyParams, htf_hours: int, htf_offset: int) -> tuple:
    return (p.donchian_n, p.atr_period, p.ema_fast, p.ema_slow, p.slope_bars, htf_hours, htf_offset)


def compute_indicators(h1: pd.DataFrame, p: StrategyParams, htf_hours: int = 4, htf_offset: int = 0) -> Indicators:
    o, h, l, c = (h1[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = to_ns(h1.index)
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.fmax.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])  # fmax ignores the NaN on bar 0
    atr = pd.Series(tr).rolling(p.atr_period).mean().to_numpy()
    hh = pd.Series(h).shift(1).rolling(p.donchian_n).max().to_numpy()
    ll = pd.Series(l).shift(1).rolling(p.donchian_n).min().to_numpy()
    rng = h - l
    clv = np.where(rng > 0, (c - l) / np.where(rng > 0, rng, 1.0), 0.5)

    # ---- HTF trend, closed bars only ----
    h4 = resample_htf(h1, htf_hours, htf_offset)
    ef = h4["close"].ewm(span=p.ema_fast, adjust=False).mean()
    es = h4["close"].ewm(span=p.ema_slow, adjust=False).mean()
    prev = ef.shift(p.slope_bars)
    tr4 = np.where((ef > es) & (ef > prev), 1, np.where((ef < es) & (ef < prev), -1, 0)).astype(np.int8)
    tr4[: p.ema_slow] = 0  # EMA warm-up
    avail = to_ns(h4.index) + htf_hours * HOUR_NS  # HTF bar close time
    j = np.searchsorted(avail, t + HOUR_NS, side="right") - 1
    trend = np.where(j >= 0, tr4[np.clip(j, 0, None)], 0).astype(np.int8)

    regime = np.full(len(c), -1, dtype=np.int8)
    rk = pd.Series(atr).rolling(720, min_periods=100).rank(pct=True).to_numpy()
    ok = np.isfinite(rk)
    regime[ok] = np.where(rk[ok] < 1 / 3, 0, np.where(rk[ok] < 2 / 3, 1, 2))

    nxt = t + HOUR_NS
    hour_next = ((nxt // HOUR_NS) % 24).astype(np.int8)
    wd_next = (((nxt // (24 * HOUR_NS)) + 3) % 7).astype(np.int8)  # 1970-01-01 was Thursday
    return Indicators(t, o, h, l, c, atr, hh, ll, clv, rng, trend, regime, hour_next, wd_next)


def entry_filter_mask(ind: Indicators, p: StrategyParams, spec: SymbolSpec) -> np.ndarray:
    """Direction-agnostic filters: volatility band, session, exhaustion-bar guard."""
    pip = spec.pip
    with np.errstate(invalid="ignore"):
        vol_ok = (ind.atr >= p.min_atr_pips * pip) & (ind.atr <= p.max_atr_pips * pip)
        sess = (ind.hour_next >= p.session_start_utc) & (ind.hour_next < p.session_end_utc)
        sess &= ~((ind.wd_next == 4) & (ind.hour_next >= p.friday_last_entry_utc))
        sess &= ind.wd_next < 5
        return np.asarray(vol_ok & sess & (ind.rng <= p.max_bar_range_atr * ind.atr), dtype=bool)


def make_signals(ind: Indicators, p: StrategyParams, spec: SymbolSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return (long, short) boolean arrays. Element i = decision at close of bar i."""
    common = entry_filter_mask(ind, p, spec)
    with np.errstate(invalid="ignore"):
        long_ = common & (ind.close > ind.hh + p.breakout_buffer_atr * ind.atr) & (ind.clv >= p.min_clv) & (ind.trend == 1)
        short_ = common & (ind.close < ind.ll - p.breakout_buffer_atr * ind.atr) & (ind.clv <= 1 - p.min_clv) & (ind.trend == -1)
    if not p.allow_long:
        long_ = np.zeros_like(long_)
    if not p.allow_short:
        short_ = np.zeros_like(short_)
    return np.asarray(long_, dtype=bool), np.asarray(short_, dtype=bool)


def explain_signal(ind: Indicators, i: int, side: int, spec: SymbolSpec) -> dict:
    """Human/machine readable reason for a signal (logged with every signal)."""
    level = ind.hh[i] if side > 0 else ind.ll[i]
    return {
        "side": "buy" if side > 0 else "sell",
        "signal_bar_time": pd.Timestamp(int(ind.t[i]), tz="UTC").isoformat(),
        "close": float(ind.close[i]),
        "breakout_level": float(level),
        "atr_pips": float(ind.atr[i] / spec.pip),
        "clv": float(ind.clv[i]),
        "bar_range_atr": float(ind.rng[i] / ind.atr[i]),
        "htf_trend": int(ind.trend[i]),
        "reason": (
            f"close {ind.close[i]:.{spec.digits}f} {'>' if side > 0 else '<'} "
            f"{'HH' if side > 0 else 'LL'} {level:.{spec.digits}f}; HTF trend {int(ind.trend[i]):+d}; "
            f"CLV {ind.clv[i]:.2f}; ATR {ind.atr[i] / spec.pip:.1f}p"
        ),
    }
