import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profx.config import RunConfig  # noqa: E402
from profx.engine import BacktestEngine, MarketData  # noqa: E402

BASE = 1.1000
WARM = 400          # flat warm-up bars so ATR/EMAs exist
SIG_ROW = WARM      # the scenario's signal bar (decision at ITS close, fill at next open)


def flat_frame(n: int = WARM, price: float = BASE, half_range: float = 0.0005, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": price, "high": price + half_range, "low": price - half_range, "close": price}, index=idx
    )


def scenario_frame(bars: list[tuple[float, float, float, float]], start: str = "2024-01-01") -> pd.DataFrame:
    """Flat warm-up + a signal bar + the given (o,h,l,c) scenario bars (bid prices)."""
    base = flat_frame(WARM + 1, start=start)
    idx = pd.date_range(base.index[0], periods=WARM + 1 + len(bars), freq="h", tz="UTC")
    rows = [(r.open, r.high, r.low, r.close) for r in base.itertuples()] + list(bars)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def sig_override(n: int, row: int, side: int):
    lg = np.zeros(n, dtype=bool)
    sh = np.zeros(n, dtype=bool)
    (lg if side > 0 else sh)[row] = True
    return {"EURUSD": (lg, sh)}


def clean_cfg(**strategy) -> RunConfig:
    """No costs except those a test enables; management disabled unless requested."""
    cfg = RunConfig(symbols=("EURUSD",))
    cfg = cfg.with_costs(spread_default_pips=1.0, slippage_entry_pips=0.0, slippage_sl_pips=0.0,
                         exec_delay_seconds=0.0, commission_per_lot_rt=0.0, swap_long_per_lot_night=0.0,
                         swap_short_per_lot_night=0.0)
    base = dict(be_r=0.0, trail_start_r=0.0, partial_pct=0.0, max_bars_in_trade=10_000,
                exit_on_trend_flip=False, close_before_weekend=False, tp_r=3.0)
    base.update(strategy)
    return cfg.with_strategy(**base)


def run_scenario(bars, side=1, cfg=None, start: str = "2024-01-01"):
    df = scenario_frame(bars, start=start)
    cfg = cfg or clean_cfg()
    md = MarketData({"EURUSD": df})
    eng = BacktestEngine(md, cfg, signal_override=sig_override(len(df), SIG_ROW, side))
    return eng.run()


@pytest.fixture
def synth():
    from profx.data import synthetic_fx
    return {s: synthetic_fx(s, years=2.5, seed=3) for s in ("EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD")}
