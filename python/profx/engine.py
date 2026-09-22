"""Event-driven multi-symbol backtest engine (hourly bars, BID data).

Order of operations for every timestamp `tk` (this ordering IS the no-look-ahead
guarantee, and it is conservative on intrabar ambiguity):

  0. Pending closes queued at the previous bar close execute at this bar's OPEN
     (also: time-based weekend flatten).
  1. Entries queued from the previous bar's CLOSE execute at this bar's OPEN,
     after all risk gates are re-evaluated with the state known right now.
  2. Intrabar SL/TP checks with the *previous* stop levels. If SL and TP could both
     have been hit inside the bar, SL wins (worst case). Gaps through stops fill at
     the gap price + slippage.
  3. End-of-bar management (extreme tracking, partial, break-even, trailing), swap,
     exit conditions to be executed at the next open, new signal generation.
     Stop changes take effect from the NEXT bar (a bar that touched +1R and then
     reversed is treated as a stop-out at the old stop).
  4. Equity mark-to-market, daily-loss and max-drawdown circuit breakers.

Cost model: spread from data (or default), entry/stop slippage, execution-delay
drift, commission per lot round-turn, swap per night (triple on Wednesday).
Shorts exit/stop against the ASK (= bid + spread).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import ConfigError, RunConfig
from .data import HOUR_NS, validate_ohlc
from .signals import Indicators, compute_indicators, explain_signal, indicator_key, make_signals, to_ns
from .specs import SymbolSpec, get_spec, round_lots_down

MIN_STOP_DIST_PIPS = 1.0   # extra distance beyond spread required to modify a stop
EPS = 1e-9


# --------------------------------------------------------------------------- market data
class MarketData:
    """Aligned multi-symbol hourly data + cached indicators."""

    def __init__(self, frames: dict[str, pd.DataFrame], htf_hours: int = 4, htf_offset: int = 0,
                 specs: dict[str, SymbolSpec] | None = None, validate: bool = True):
        if not frames:
            raise ConfigError("no data")
        self.htf_hours, self.htf_offset = htf_hours, htf_offset
        self.symbols = list(frames)
        self.specs = {s: (specs or {}).get(s) or get_spec(s) for s in self.symbols}
        self.frames = frames
        if validate:
            for s, df in frames.items():
                validate_ohlc(df, s)
        self.t = {s: to_ns(df.index) for s, df in frames.items()}
        self.timeline = np.unique(np.concatenate(list(self.t.values())))
        self.maps = {s: pd.Index(self.t[s]).get_indexer(self.timeline).astype(np.int64) for s in self.symbols}
        self.o = {s: df["open"].to_numpy(float) for s, df in frames.items()}
        self.h = {s: df["high"].to_numpy(float) for s, df in frames.items()}
        self.l = {s: df["low"].to_numpy(float) for s, df in frames.items()}
        self.c = {s: df["close"].to_numpy(float) for s, df in frames.items()}
        self.spread_pts = {s: (df["spread"].to_numpy(float) if "spread" in df else np.full(len(df), np.nan))
                           for s, df in frames.items()}
        self.quote_usd = {s: self._quote_usd(s) for s in self.symbols}
        self._ind_cache: dict[tuple, Indicators] = {}

    def _quote_usd(self, s: str) -> np.ndarray:
        sp = self.specs[s]
        n = len(self.c[s])
        if sp.quote == "USD":
            return np.ones(n)
        if sp.base == "USD":
            return 1.0 / self.c[s]
        for cs, inv in ((f"USD{sp.quote}", True), (f"{sp.quote}USD", False)):
            if cs in self.frames:
                ser = self.frames[cs]["close"].reindex(self.frames[s].index, method="ffill").bfill().to_numpy(float)
                return 1.0 / ser if inv else ser
        raise ConfigError(f"{s}: need USD{sp.quote} or {sp.quote}USD data for account-currency conversion (USD account)")

    def indicators(self, s: str, p) -> Indicators:
        key = (s,) + indicator_key(p, self.htf_hours, self.htf_offset)
        if key not in self._ind_cache:
            self._ind_cache[key] = compute_indicators(self.frames[s], p, self.htf_hours, self.htf_offset)
        return self._ind_cache[key]

    def slice_ns(self, start: pd.Timestamp | None, end: pd.Timestamp | None) -> tuple[int, int]:
        def _ns(ts: pd.Timestamp) -> int:
            ts = pd.Timestamp(ts)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            return int(to_ns(pd.DatetimeIndex([ts]))[0])

        a = _ns(start) if start is not None else int(self.timeline[0])
        b = _ns(end) if end is not None else int(self.timeline[-1]) + 1
        return a, b


# --------------------------------------------------------------------------- state
@dataclass
class Position:
    symbol: str
    side: int
    lots: float
    lots_initial: float
    entry_ns: int
    entry_row: int
    entry_price: float
    sl: float
    tp: float
    r_dist: float
    risk_money: float
    extreme: float
    signal_id: str
    regime: int
    atr_pips: float
    bars_held: int = 0
    be_done: bool = False
    partial_done: bool = False
    pending_close: str | None = None
    gross: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    sl_initial: float = 0.0


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    signals: pd.DataFrame
    rejections: dict
    halted: str | None
    halted_at: pd.Timestamp | None
    exposure: float
    initial_balance: float
    stop_loosen_attempts: int
    n_halts: int = 0
    cfg: RunConfig = field(repr=False, default=None)


# --------------------------------------------------------------------------- engine
class BacktestEngine:
    def __init__(self, market: MarketData, cfg: RunConfig,
                 signal_override: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
                 window: tuple[pd.Timestamp | None, pd.Timestamp | None] = (None, None)):
        self.m, self.cfg = market, cfg
        self.sp_, self.rk, self.co = cfg.strategy, cfg.risk, cfg.costs
        self.override = signal_override
        self.win = market.slice_ns(*window)

    # ------------------------------------------------------------- helpers
    def _spread_price(self, s: str) -> np.ndarray:
        spec = self.m.specs[s]
        pts = self.m.spread_pts[s]
        base = np.where(np.isnan(pts), self.co.spread_default_pips * spec.pip, pts * spec.point)
        return base * self.co.spread_mult

    def _delay_cost(self, atr: float) -> float:
        return 0.4 * atr * math.sqrt(max(self.co.exec_delay_seconds, 0.0) / 3600.0)

    def _pnl(self, pos: Position, exit_price: float, lots: float, conv: float) -> float:
        spec = self.m.specs[pos.symbol]
        return pos.side * (exit_price - pos.entry_price) * lots * spec.contract_size * conv

    # ------------------------------------------------------------- main loop
    def run(self) -> BacktestResult:
        m, sp_, rk, co = self.m, self.sp_, self.rk, self.co
        syms = m.symbols
        T = m.timeline
        lo, hi = self.win
        k0, k1 = int(np.searchsorted(T, lo, "left")), int(np.searchsorted(T, hi, "left"))
        if k1 - k0 < 2:
            raise ConfigError("empty backtest window")

        ind = {s: m.indicators(s, sp_) for s in syms}
        sigs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for s in syms:
            sigs[s] = self.override[s] if self.override and s in self.override else make_signals(ind[s], sp_, m.specs[s])
        spr = {s: self._spread_price(s) for s in syms}

        balance = self.cfg.initial_balance
        positions: dict[str, Position] = {}
        pending: dict[str, tuple[int, int, dict]] = {}   # symbol -> (signal_row, side, explanation)
        last_row: dict[str, int] = {}
        last_exit_ns: dict[str, int] = {}
        trades: list[dict] = []
        sig_log: list[dict] = []
        rej: Counter = Counter()
        eq_t: list[int] = []
        eq_v: list[float] = []
        bars_with_pos = 0
        day_key, day_start_eq, daily_trades = None, balance, 0
        daily_blocked = False
        halted: str | None = None
        halted_at: int | None = None
        peak_eq = balance
        loosen = 0
        n_halts = 0

        def conv_at(s: str, i: int) -> float:
            return float(m.quote_usd[s][i])

        def floating() -> float:
            tot = 0.0
            for s, ps in positions.items():
                i = last_row[s]
                px = m.c[s][i] if ps.side > 0 else m.c[s][i] + spr[s][i]
                tot += self._pnl(ps, px, ps.lots, conv_at(s, i))
            return tot

        def close_position(ps: Position, price: float, i: int, ns: int, reason: str, lots: float | None = None) -> None:
            nonlocal balance
            lots_c = ps.lots if lots is None else lots
            conv = conv_at(ps.symbol, i)
            gross = self._pnl(ps, price, lots_c, conv)
            comm = co.commission_per_lot_rt * co.commission_mult * lots_c
            ps.gross += gross
            ps.commission += comm
            balance += gross - comm
            if lots is None or lots_c >= ps.lots - EPS:
                net = ps.gross - ps.commission + ps.swap
                trades.append({
                    "symbol": ps.symbol, "side": "buy" if ps.side > 0 else "sell",
                    "entry_time": pd.Timestamp(ps.entry_ns, tz="UTC"), "exit_time": pd.Timestamp(ns, tz="UTC"),
                    "entry_price": ps.entry_price, "exit_price": price, "sl_initial": ps.sl_initial,
                    "lots": ps.lots_initial, "gross": ps.gross, "commission": ps.commission, "swap": ps.swap,
                    "pnl": net, "risk_money": ps.risk_money, "r": net / ps.risk_money if ps.risk_money > 0 else 0.0,
                    "exit_reason": reason, "bars_held": ps.bars_held, "regime": ps.regime,
                    "atr_pips": ps.atr_pips, "signal_id": ps.signal_id,
                })
                last_exit_ns[ps.symbol] = ns
                positions.pop(ps.symbol, None)
            else:
                ps.lots = round(ps.lots - lots_c, 8)

        def equity() -> float:
            return balance + floating()

        for k in range(k0, k1):
            tk = int(T[k])
            dk = tk // (24 * HOUR_NS)
            if dk != day_key:
                day_key, day_start_eq, daily_trades, daily_blocked = dk, equity(), 0, False
            wd = int((((tk // (24 * HOUR_NS)) + 3) % 7))
            hr = int((tk // HOUR_NS) % 24)

            # ---- step 0: pending closes at OPEN (+ weekend flatten)
            for s, ps in list(positions.items()):
                i = int(m.maps[s][k])
                if i < 0:
                    continue
                reason = ps.pending_close
                if reason is None and sp_.close_before_weekend and wd == 4 and hr >= sp_.weekend_close_hour_utc:
                    reason = "weekend_close"
                if reason:
                    px = m.o[s][i] if ps.side > 0 else m.o[s][i] + spr[s][i]
                    close_position(ps, px, i, tk, reason)
                    last_row[s] = i

            # ---- step 1: entries at OPEN
            for s in list(pending):
                i = int(m.maps[s][k])
                if i < 0:
                    continue
                sig_row, side, expl = pending.pop(s)
                sid = f"{s}-{expl['signal_bar_time']}-{expl['side']}"
                outcome = self._try_enter(s, i, side, sig_row, expl, sid, k, tk, ind[s], spr[s], positions, balance,
                                          equity, halted, daily_blocked, daily_trades, last_exit_ns, rej, last_row, spr)
                if isinstance(outcome, Position):
                    positions[s] = outcome
                    last_row[s] = i
                    daily_trades += 1
                    sig_log.append({**expl, "signal_id": sid, "outcome": "taken"})
                else:
                    sig_log.append({**expl, "signal_id": sid, "outcome": f"rejected:{outcome}"})

            # ---- step 2: intrabar stops / targets (previous stop levels)
            for s, ps in list(positions.items()):
                i = int(m.maps[s][k])
                if i < 0:
                    continue
                last_row[s] = i
                hit = self._intrabar(ps, s, i, spr[s][i])
                if hit is not None:
                    close_position(ps, hit[0], i, tk, hit[1])

            # ---- step 3: end-of-bar management, swap, next-open exits, signals
            for s, ps in list(positions.items()):
                i = int(m.maps[s][k])
                if i < 0:
                    continue
                loosen += self._manage(ps, s, i, spr[s][i], ind[s], close_position, tk)
                if ps.symbol in positions:  # still open (partial keeps it open)
                    if hr == 21:
                        per = co.swap_long_per_lot_night if ps.side > 0 else co.swap_short_per_lot_night
                        ps.swap += per * co.swap_mult * ps.lots * (3 if wd == 2 else 1)
                        balance += per * co.swap_mult * ps.lots * (3 if wd == 2 else 1)
                    ps.bars_held += 1
                    if ps.pending_close is None:
                        if ps.bars_held >= sp_.max_bars_in_trade:
                            ps.pending_close = "time_exit"
                        elif sp_.exit_on_trend_flip and int(ind[s].trend[i]) == -ps.side:
                            ps.pending_close = "trend_flip"
            for s in syms:
                i = int(m.maps[s][k])
                if i < 0 or s in positions or s in pending:
                    continue
                lg, sh = sigs[s][0][i], sigs[s][1][i]
                if lg or sh:
                    side = 1 if lg else -1
                    pending[s] = (i, side, explain_signal(ind[s], i, side, m.specs[s]))

            # ---- step 4: equity + circuit breakers
            for s in syms:
                i = int(m.maps[s][k])
                if i >= 0:
                    last_row[s] = i
            eq = equity()
            eq_t.append(tk)
            eq_v.append(eq)
            if positions:
                bars_with_pos += 1
            peak_eq = max(peak_eq, eq)
            if halted is None and (peak_eq - eq) / peak_eq * 100.0 >= rk.max_drawdown_pct:
                halted, halted_at = "max_drawdown", tk
                n_halts += 1
                for ps in positions.values():
                    ps.pending_close = ps.pending_close or "max_drawdown"
            elif (halted is not None and rk.dd_halt_cooldown_days > 0 and not positions
                  and tk - int(halted_at) >= rk.dd_halt_cooldown_days * 24 * HOUR_NS):
                halted, peak_eq = None, eq      # research-only: simulated operator reset
            if not daily_blocked and (eq / day_start_eq - 1.0) * 100.0 <= -rk.daily_loss_limit_pct:
                daily_blocked = True
                if rk.flatten_on_daily_loss:
                    for ps in positions.values():
                        ps.pending_close = ps.pending_close or "daily_loss_limit"

        # ---- end of window: flatten at last bar close
        for s, ps in list(positions.items()):
            i = last_row[s]
            px = m.c[s][i] if ps.side > 0 else m.c[s][i] + spr[s][i]
            close_position(ps, px, i, int(m.t[s][i]) + HOUR_NS, "window_end")

        tr = pd.DataFrame(trades)
        eq_series = pd.Series(eq_v, index=pd.DatetimeIndex(pd.to_datetime(eq_t, unit="ns", utc=True), name="time"), name="equity")
        return BacktestResult(
            trades=tr, equity=eq_series, signals=pd.DataFrame(sig_log), rejections=dict(rej), halted=halted,
            halted_at=pd.Timestamp(halted_at, tz="UTC") if halted_at else None,
            exposure=bars_with_pos / max(k1 - k0, 1), initial_balance=self.cfg.initial_balance,
            stop_loosen_attempts=loosen, n_halts=n_halts, cfg=self.cfg,
        )

    # ------------------------------------------------------------- entry
    def _try_enter(self, s, i, side, sig_row, expl, sid, k, tk, ind, spr, positions, balance, equity_fn, halted,
                   daily_blocked, daily_trades, last_exit_ns, rej, last_row, spr_all):
        m, sp_, rk, co = self.m, self.sp_, self.rk, self.co
        spec = m.specs[s]
        T_s = m.t[s]

        def no(reason: str) -> str:
            rej[reason] += 1
            return reason

        if halted:
            return no("halted")
        if daily_blocked:
            return no("daily_loss_block")
        if sig_row != i - 1 or T_s[i] - T_s[i - 1] != HOUR_NS:
            return no("gap_or_stale")
        if s in positions:
            return no("symbol_busy")
        if len(positions) >= rk.max_open_positions:
            return no("max_positions")
        if daily_trades >= rk.max_daily_trades:
            return no("max_daily_trades")
        if s in last_exit_ns and tk - last_exit_ns[s] < rk.cooldown_bars * HOUR_NS:
            return no("cooldown")

        atr = float(ind.atr[sig_row])
        D = sp_.sl_atr_mult * atr
        sp = float(spr[i])
        if sp / spec.pip > rk.max_spread_pips:
            return no("spread_abs")
        if sp > rk.max_spread_frac_of_sl * D:
            return no("spread_rel")

        slip = co.slippage_entry_pips * co.slippage_mult * spec.pip + self._delay_cost(atr)
        o = float(m.o[s][i])
        fill = o + sp + slip if side > 0 else o - slip
        sl = fill - side * D
        tp = fill + side * sp_.tp_r * D if sp_.tp_r > 0 else 0.0

        eq = equity_fn()
        basis = eq if rk.risk_basis == "equity" else balance
        conv = float(m.quote_usd[s][i])
        val = spec.contract_size * conv
        comm = co.commission_per_lot_rt * co.commission_mult
        lots = round_lots_down(basis * rk.risk_pct / 100.0 / (D * val + comm), spec)
        if lots < spec.vol_min - EPS:
            return no("lot_below_min")
        risk_money = lots * (D * val + comm)

        open_risk = 0.0
        for ps in positions.values():
            pv = m.specs[ps.symbol].contract_size * float(m.quote_usd[ps.symbol][last_row[ps.symbol]])
            open_risk += max(0.0, ps.side * (ps.entry_price - ps.sl)) * pv * ps.lots
        if open_risk + risk_money > rk.max_total_risk_pct / 100.0 * eq:
            return no("max_total_risk")

        exp: Counter = Counter()
        for ps in positions.values():
            sp0 = m.specs[ps.symbol]
            exp[(sp0.base, ps.side)] += 1
            exp[(sp0.quote, -ps.side)] += 1
        if exp[(spec.base, side)] + 1 > rk.max_same_ccy_dir or exp[(spec.quote, -side)] + 1 > rk.max_same_ccy_dir:
            return no("currency_exposure")

        base_usd = 1.0 if spec.base == "USD" else (o if spec.quote == "USD" else o * conv)
        margin_new = lots * spec.contract_size * base_usd / rk.leverage
        margin_used = 0.0
        for ps in positions.values():
            sp0 = m.specs[ps.symbol]
            px = float(m.c[ps.symbol][last_row[ps.symbol]])
            b_usd = 1.0 if sp0.base == "USD" else (px if sp0.quote == "USD" else px * float(m.quote_usd[ps.symbol][last_row[ps.symbol]]))
            margin_used += ps.lots * sp0.contract_size * b_usd / rk.leverage
        if eq - (margin_used + margin_new) < rk.min_free_margin_pct / 100.0 * eq:
            return no("margin")

        return Position(
            symbol=s, side=side, lots=lots, lots_initial=lots, entry_ns=tk, entry_row=i, entry_price=fill, sl=sl,
            tp=tp, r_dist=D, risk_money=risk_money, extreme=fill if side > 0 else o + sp, signal_id=sid,
            regime=int(ind.regime[sig_row]), atr_pips=atr / spec.pip, sl_initial=sl,
        )

    # ------------------------------------------------------------- intrabar
    def _intrabar(self, ps: Position, s: str, i: int, sp: float):
        m, co = self.m, self.co
        spec = m.specs[s]
        slip = co.slippage_sl_pips * co.slippage_mult * spec.pip
        o, h, l = m.o[s][i], m.h[s][i], m.l[s][i]
        if ps.side > 0:
            if o <= ps.sl:
                return o - slip, "sl_gap"
            if l <= ps.sl:
                return ps.sl - slip, "sl"
            if ps.tp > 0 and h >= ps.tp:
                return max(o, ps.tp), "tp"
        else:
            ao, ah, al = o + sp, h + sp, l + sp
            if ao >= ps.sl:
                return ao + slip, "sl_gap"
            if ah >= ps.sl:
                return ps.sl + slip, "sl"
            if ps.tp > 0 and al <= ps.tp:
                return min(ao, ps.tp), "tp"
        return None

    # ------------------------------------------------------------- management
    def _manage(self, ps: Position, s: str, i: int, sp: float, ind: Indicators, close_fn, tk: int) -> int:
        """End-of-bar management. Returns 1 if the stop-never-loosens invariant was violated (must stay 0)."""
        m, sp_ = self.m, self.sp_
        spec = m.specs[s]
        h, l, c = m.h[s][i], m.l[s][i], m.c[s][i]
        side, R = ps.side, ps.r_dist
        ps.extreme = max(ps.extreme, h) if side > 0 else min(ps.extreme, l + sp)
        fav = (ps.extreme - ps.entry_price) if side > 0 else (ps.entry_price - ps.extreme)
        ref = c if side > 0 else c + sp                      # price the stop is compared to
        min_dist = sp + MIN_STOP_DIST_PIPS * spec.pip
        sl_before = ps.sl

        def tighten(new_sl: float) -> None:
            """Apply only IMPROVEMENTS that keep the stop clear of the market."""
            if side > 0:
                if new_sl > ps.sl + EPS and new_sl < ref - min_dist:
                    ps.sl = new_sl
            else:
                if new_sl < ps.sl - EPS and new_sl > ref + min_dist:
                    ps.sl = new_sl

        if sp_.partial_pct > 0 and not ps.partial_done and fav >= sp_.partial_r * R - EPS:
            part = round_lots_down(ps.lots * sp_.partial_pct, spec)
            if part >= spec.vol_min - EPS and ps.lots - part >= spec.vol_min - EPS:
                level = ps.entry_price + side * sp_.partial_r * R
                close_fn(ps, level, i, tk + HOUR_NS, "partial", lots=part)
            ps.partial_done = True
        if sp_.be_r > 0 and not ps.be_done and fav >= sp_.be_r * R - EPS:
            cand = ps.entry_price + side * sp_.be_offset_pips * spec.pip
            if (side > 0 and cand > ps.sl) or (side < 0 and cand < ps.sl):
                tighten(cand)
            ps.be_done = True
        if sp_.trail_start_r > 0 and fav >= sp_.trail_start_r * R - EPS:
            atr = ind.atr[i]
            if np.isfinite(atr):
                tighten(ps.extreme - side * sp_.trail_atr_mult * atr)
        # invariant: a stop must never move so that risk INCREASES
        worse = (side > 0 and ps.sl < sl_before - EPS) or (side < 0 and ps.sl > sl_before + EPS)
        return 1 if worse else 0
