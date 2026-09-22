"""Configuration for the ProFX research/backtest stack.

Defaults here MUST mirror the MQL5 EA inputs (mt5/MQL5/Include/ProFX/Config.mqh)
and the Pine script inputs, otherwise backtests and live trading diverge.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised for invalid or inconsistent configuration."""


@dataclass(frozen=True)
class StrategyParams:
    # --- signal (evaluated on CLOSED execution-timeframe bars) ---
    donchian_n: int = 24            # breakout lookback (bars, excludes signal bar)
    atr_period: int = 14            # SMA-of-True-Range (same as MT5 iATR)
    ema_fast: int = 50              # higher-timeframe trend EMAs (closed HTF bars only)
    ema_slow: int = 200
    slope_bars: int = 6             # EMA_fast(now) must exceed EMA_fast(now - slope_bars)
    breakout_buffer_atr: float = 0.05
    min_clv: float = 0.60           # close-location value in top 60%..100% of bar range
    max_bar_range_atr: float = 2.5  # reject exhaustion bars
    min_atr_pips: float = 5.0
    max_atr_pips: float = 50.0
    session_start_utc: int = 7      # entry-bar open hour in [start, end)
    session_end_utc: int = 19
    friday_last_entry_utc: int = 17
    allow_long: bool = True
    allow_short: bool = True
    # --- stops / exits ---
    sl_atr_mult: float = 1.5
    tp_r: float = 3.0
    be_r: float = 1.0               # 0 disables
    be_offset_pips: float = 1.0
    trail_start_r: float = 1.5      # 0 disables
    trail_atr_mult: float = 2.0
    partial_r: float = 1.5
    partial_pct: float = 0.0        # 0 disables (fraction of position, 0..0.9)
    max_bars_in_trade: int = 72
    exit_on_trend_flip: bool = True
    close_before_weekend: bool = True
    weekend_close_hour_utc: int = 20

    def __post_init__(self) -> None:
        def need(cond: bool, msg: str) -> None:
            if not cond:
                raise ConfigError(f"StrategyParams: {msg}")

        need(self.donchian_n >= 2, "donchian_n >= 2")
        need(self.atr_period >= 2, "atr_period >= 2")
        need(1 <= self.ema_fast < self.ema_slow, "need 1 <= ema_fast < ema_slow")
        need(self.slope_bars >= 1, "slope_bars >= 1")
        need(0.0 <= self.breakout_buffer_atr <= 2.0, "breakout_buffer_atr in [0,2]")
        need(0.5 <= self.min_clv <= 1.0, "min_clv in [0.5,1]")
        need(self.max_bar_range_atr > 0, "max_bar_range_atr > 0")
        need(0 < self.min_atr_pips < self.max_atr_pips, "0 < min_atr_pips < max_atr_pips")
        need(0 <= self.session_start_utc < self.session_end_utc <= 24, "session hours")
        need(0 <= self.friday_last_entry_utc <= 24, "friday_last_entry_utc")
        need(self.sl_atr_mult > 0, "sl_atr_mult > 0")
        need(self.tp_r >= 0, "tp_r >= 0 (0 disables TP)")
        need(self.be_r >= 0 and self.trail_start_r >= 0, "be_r/trail_start_r >= 0")
        need(self.trail_atr_mult > 0, "trail_atr_mult > 0")
        need(0.0 <= self.partial_pct <= 0.9, "partial_pct in [0,0.9]")
        need(self.partial_r > 0, "partial_r > 0")
        need(self.max_bars_in_trade >= 1, "max_bars_in_trade >= 1")
        need(0 <= self.weekend_close_hour_utc <= 23, "weekend_close_hour_utc")


@dataclass(frozen=True)
class RiskParams:
    risk_pct: float = 0.5                 # % of equity risked per trade
    risk_basis: str = "equity"            # "equity" | "balance"
    max_open_positions: int = 3
    max_positions_per_symbol: int = 1
    max_total_risk_pct: float = 1.5       # sum of open risk-to-SL, % equity
    max_daily_trades: int = 4
    cooldown_bars: int = 4                # after a close on that symbol
    daily_loss_limit_pct: float = 2.0
    max_drawdown_pct: float = 8.0         # from peak equity -> flatten + halt
    max_same_ccy_dir: int = 2             # correlation/currency-exposure cap
    max_spread_pips: float = 3.0
    max_spread_frac_of_sl: float = 0.15
    min_free_margin_pct: float = 50.0     # free margin after trade, % of equity
    leverage: float = 30.0
    flatten_on_daily_loss: bool = True
    # LIVE behaviour = 0 (breaker latches until a human resets it). For RESEARCH set N>0 to
    # model an operator restarting after N days with the peak reset; otherwise a single early
    # drawdown ends the whole backtest and hides the rest of the sample.
    dd_halt_cooldown_days: int = 0

    def __post_init__(self) -> None:
        if not (0 < self.risk_pct <= 2.0):
            raise ConfigError("risk_pct must be in (0, 2] (hard safety cap)")
        if self.risk_basis not in ("equity", "balance"):
            raise ConfigError("risk_basis must be 'equity' or 'balance'")
        if self.max_open_positions < 1 or self.max_positions_per_symbol < 1:
            raise ConfigError("position limits must be >= 1")
        if self.max_total_risk_pct < self.risk_pct:
            raise ConfigError("max_total_risk_pct must be >= risk_pct")
        if not (0 < self.daily_loss_limit_pct < 100 and 0 < self.max_drawdown_pct < 100):
            raise ConfigError("loss limits must be in (0,100)")
        if self.leverage <= 0:
            raise ConfigError("leverage must be > 0")
        if self.dd_halt_cooldown_days < 0:
            raise ConfigError("dd_halt_cooldown_days must be >= 0")


@dataclass(frozen=True)
class CostParams:
    spread_default_pips: float = 1.0      # used only if data has no 'spread' column
    spread_mult: float = 1.0              # stress knob
    slippage_entry_pips: float = 0.2
    slippage_sl_pips: float = 0.5
    slippage_mult: float = 1.0            # stress knob
    exec_delay_seconds: float = 1.0       # adverse drift model, see engine._delay_cost
    commission_per_lot_rt: float = 7.0    # account ccy, round turn
    commission_mult: float = 1.0
    swap_long_per_lot_night: float = -3.0  # ASSUMPTION - set from your broker
    swap_short_per_lot_night: float = -3.0
    swap_mult: float = 1.0


@dataclass(frozen=True)
class RunConfig:
    symbols: tuple[str, ...] = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD")
    initial_balance: float = 10_000.0
    htf_hours: int = 4
    htf_offset_hours: int = 0             # H4 bar alignment vs UTC (broker dependent)
    strategy: StrategyParams = field(default_factory=StrategyParams)
    risk: RiskParams = field(default_factory=RiskParams)
    costs: CostParams = field(default_factory=CostParams)

    def with_strategy(self, **kw: Any) -> "RunConfig":
        return replace(self, strategy=replace(self.strategy, **kw))

    def with_risk(self, **kw: Any) -> "RunConfig":
        return replace(self, risk=replace(self.risk, **kw))

    def with_costs(self, **kw: Any) -> "RunConfig":
        return replace(self, costs=replace(self.costs, **kw))


def _build(cls: type, table: dict[str, Any], name: str):
    valid = {f.name for f in fields(cls)}
    unknown = set(table) - valid
    if unknown:
        raise ConfigError(f"[{name}] unknown keys: {sorted(unknown)}")
    return cls(**table)


def load_config(path: str | Path) -> RunConfig:
    """Load a TOML config. Unknown keys are rejected (typos must not pass silently)."""
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    run = dict(raw.get("run", {}))
    if "symbols" in run:
        run["symbols"] = tuple(run["symbols"])
    valid_run = {f.name for f in fields(RunConfig)} - {"strategy", "risk", "costs"}
    unknown = set(run) - valid_run
    if unknown:
        raise ConfigError(f"[run] unknown keys: {sorted(unknown)}")
    return RunConfig(
        strategy=_build(StrategyParams, dict(raw.get("strategy", {})), "strategy"),
        risk=_build(RiskParams, dict(raw.get("risk", {})), "risk"),
        costs=_build(CostParams, dict(raw.get("costs", {})), "costs"),
        **run,
    )
