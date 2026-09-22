"""Symbol specifications.

These are TYPICAL retail values. Real brokers differ (digits, lot step, min lot,
contract size, stop level). The MT5 EA reads the real values from the terminal;
for backtests, override SPECS from your broker via `scripts/export_mt5_history.py`
(which also dumps symbol specs) before trusting results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import ConfigError


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str
    base: str
    quote: str
    digits: int
    contract_size: float = 100_000.0
    vol_min: float = 0.01
    vol_max: float = 50.0
    vol_step: float = 0.01

    @property
    def point(self) -> float:
        return 10.0 ** (-self.digits)

    @property
    def pip(self) -> float:
        # 5-digit / 3-digit quotes: 1 pip = 10 points. 4/2-digit: 1 pip = 1 point.
        return self.point * (10.0 if self.digits in (3, 5) else 1.0)


def _mk(sym: str, digits: int) -> SymbolSpec:
    return SymbolSpec(symbol=sym, base=sym[:3], quote=sym[3:6], digits=digits)


SPECS: dict[str, SymbolSpec] = {
    s: _mk(s, 3 if s.endswith("JPY") else 5)
    for s in (
        "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCAD", "USDCHF",
        "EURJPY", "GBPJPY", "EURGBP", "EURCHF", "AUDJPY", "EURAUD", "GBPCHF",
    )
}


def get_spec(symbol: str) -> SymbolSpec:
    try:
        return SPECS[symbol]
    except KeyError as exc:
        raise ConfigError(f"No SymbolSpec for {symbol}; add it to specs.SPECS") from exc


def round_lots_down(lots: float, spec: SymbolSpec) -> float:
    """Floor to lot step (never round UP: that would exceed the risk budget)."""
    if not math.isfinite(lots) or lots <= 0:
        return 0.0
    steps = math.floor(lots / spec.vol_step + 1e-9)
    return round(min(steps * spec.vol_step, spec.vol_max), 8)
