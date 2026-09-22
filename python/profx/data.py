"""Data handling.

Conventions
-----------
* All frames use a tz-aware UTC DatetimeIndex = bar OPEN time.
* Prices are BID (MT5 default). `spread` column (optional) is in POINTS.
* Bars are hourly (execution timeframe). Higher-timeframe (HTF) frames are derived
  ONLY by resampling those bars, and an HTF bar becomes usable only at its close time.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .specs import get_spec

HOUR_NS = 3_600 * 1_000_000_000
OHLC = ["open", "high", "low", "close"]


class DataError(ValueError):
    pass


def validate_ohlc(df: pd.DataFrame, symbol: str = "?") -> list[str]:
    """Hard errors raise DataError; soft issues are returned as warnings."""
    warnings: list[str] = []
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise DataError(f"{symbol}: index must be tz-aware DatetimeIndex (UTC)")
    if not df.index.is_monotonic_increasing:
        raise DataError(f"{symbol}: index not sorted")
    if df.index.has_duplicates:
        raise DataError(f"{symbol}: duplicate timestamps")
    missing = [c for c in OHLC if c not in df.columns]
    if missing:
        raise DataError(f"{symbol}: missing columns {missing}")
    if df[OHLC].isna().any().any():
        raise DataError(f"{symbol}: NaN in OHLC")
    if (df[OHLC] <= 0).any().any():
        raise DataError(f"{symbol}: non-positive prices")
    bad = (df["high"] < df[["open", "close", "low"]].max(axis=1) - 1e-12) | (
        df["low"] > df[["open", "close", "high"]].min(axis=1) + 1e-12
    )
    if bad.any():
        raise DataError(f"{symbol}: {int(bad.sum())} bars violate high/low bounds")
    gaps = df.index.to_series().diff().dropna()
    big = gaps[gaps > pd.Timedelta(hours=100)]
    if len(big):
        warnings.append(f"{symbol}: {len(big)} gaps > 100h (missing data?)")
    return warnings


def load_csv(
    path: str | Path,
    server_tz: str | None = None,
    server_utc_offset_hours: float | None = None,
) -> pd.DataFrame:
    """Load OHLC CSV. Required: a time column + open/high/low/close.

    Time handling (pick ONE, otherwise data is assumed to already be UTC):
      server_tz="Europe/Athens"   -> DST-aware conversion (typical 'NY-close' broker
                                     servers follow US DST; Athens is only an
                                     approximation around DST switch weeks - verify!)
      server_utc_offset_hours=2   -> fixed offset (ignores DST; will be wrong half the year
                                     for DST-following servers)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in ("time", "datetime", "date", "timestamp") if c in df.columns), None)
    if tcol is None:
        raise DataError(f"{path}: no time column")
    if np.issubdtype(df[tcol].dtype, np.number):
        idx = pd.to_datetime(df[tcol], unit="s")
    else:
        idx = pd.to_datetime(df[tcol])
    if server_tz and server_utc_offset_hours is not None:
        raise DataError("Pass either server_tz or server_utc_offset_hours, not both")
    if server_tz:
        idx = idx.dt.tz_localize(server_tz, ambiguous="infer", nonexistent="shift_forward").dt.tz_convert("UTC")
    elif server_utc_offset_hours is not None:
        idx = (idx - pd.Timedelta(hours=server_utc_offset_hours)).dt.tz_localize("UTC")
    else:
        idx = idx.dt.tz_localize("UTC") if idx.dt.tz is None else idx.dt.tz_convert("UTC")
    df.index = pd.DatetimeIndex(idx, name="time")
    df = df.drop(columns=[tcol]).sort_index()
    keep = OHLC + [c for c in ("spread", "tick_volume") if c in df.columns]
    return df[keep].astype(float)


def resample_htf(h1: pd.DataFrame, hours: int = 4, offset_hours: int = 0) -> pd.DataFrame:
    """Aggregate hourly bars into HTF bars. Index = HTF bar OPEN time."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    out = h1[OHLC].resample(f"{hours}h", offset=f"{offset_hours}h", label="left", closed="left").agg(agg)
    return out.dropna()


def synthetic_fx(
    symbol: str,
    start: str = "2019-01-01",
    years: float = 6.0,
    seed: int = 0,
    trend_strength: float = 0.35,
) -> pd.DataFrame:
    """Regime-switching synthetic hourly FX data. FOR TESTING PIPELINES ONLY.

    Performance measured on this data says NOTHING about real markets.
    trend_strength=0 produces a driftless random walk (a good null hypothesis:
    a sound backtester must not find a stable edge there).
    """
    spec = get_spec(symbol)
    rng = np.random.default_rng((zlib.crc32(symbol.encode()) + 7919 * seed) % (2**32))
    idx = pd.date_range(start, periods=int(years * 365.25 * 24), freq="h", tz="UTC")
    wd, hr = idx.weekday, idx.hour
    keep = ~(((wd == 4) & (hr >= 22)) | (wd == 5) | ((wd == 6) & (hr < 22)))
    idx = idx[keep]
    n = len(idx)
    hr = idx.hour.to_numpy()

    base = {"EURUSD": 1.10, "GBPUSD": 1.30, "AUDUSD": 0.70, "NZDUSD": 0.65, "USDJPY": 110.0,
            "USDCAD": 1.30, "USDCHF": 0.95, "EURJPY": 130.0, "GBPJPY": 145.0}.get(symbol, 1.0)
    sigma_h = 0.00055  # relative hourly vol
    seasonal = np.where((hr >= 7) & (hr < 17), 1.35, np.where((hr >= 22) | (hr < 6), 0.7, 1.0))
    # vol clustering (AR(1) in log-vol) + regimes
    lv = np.zeros(n)
    eps = rng.normal(0, 0.08, n)
    for i in range(1, n):
        lv[i] = 0.98 * lv[i - 1] + eps[i]
    volmult = np.exp(lv - lv.var() / 2)
    regime = np.zeros(n, dtype=int)  # -1 down, 0 range, +1 up
    cur = 0
    switch = rng.random(n) < 1 / 240
    picks = rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3])
    for i in range(n):
        if switch[i]:
            cur = int(picks[i])
        regime[i] = cur
    drift = regime * trend_strength * sigma_h * 0.25
    vol = sigma_h * seasonal * volmult
    sub = 12
    inc = rng.normal(0.0, 1.0, (n, sub)) * (vol[:, None] / np.sqrt(sub)) + (drift[:, None] / sub)
    # weekend gaps
    dt = idx.to_series().diff().dt.total_seconds().to_numpy()
    gap_rows = np.where(dt > 3600 * 6)[0]
    inc[gap_rows, 0] += rng.normal(0, 2.0 * sigma_h, len(gap_rows))
    ends = np.cumsum(inc.sum(axis=1))
    opens = np.concatenate([[0.0], ends[:-1]])
    path = opens[:, None] + np.cumsum(inc, axis=1)
    highs = np.maximum(opens, path.max(axis=1))
    lows = np.minimum(opens, path.min(axis=1))
    o, h, l, c = (base * np.exp(x) for x in (opens, highs, lows, ends))
    spread_pips = np.where((hr >= 21) | (hr < 1), 2.2, np.where((hr >= 7) & (hr < 17), 0.7, 1.1))
    spread_pts = np.round(spread_pips * (spec.pip / spec.point) * rng.uniform(0.85, 1.25, n))
    df = pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c,
         "tick_volume": rng.integers(200, 3000, n).astype(float), "spread": spread_pts},
        index=pd.DatetimeIndex(idx, name="time"),
    )
    df[OHLC] = df[OHLC].round(spec.digits)
    # rounding can break bounds by 1 point; repair
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df
