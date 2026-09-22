"""Look-ahead / repainting guarantees.

Strategy: a causal computation must give IDENTICAL results for rows <= j whether or not
data after j exists, and whether or not that future data is corrupted.
"""
import numpy as np
import pandas as pd
import pytest

from profx.config import RunConfig, StrategyParams
from profx.data import synthetic_fx
from profx.engine import BacktestEngine, MarketData
from profx.signals import compute_indicators, make_signals
from profx.specs import get_spec

FIELDS = ["atr", "hh", "ll", "clv", "rng", "trend", "hour_next", "wd_next"]


def assert_same(a, b, upto):
    for f in FIELDS:
        x, y = getattr(a, f)[: upto + 1], getattr(b, f)[: upto + 1]
        assert np.allclose(x, y, equal_nan=True), f"{f} differs (look-ahead!)"


@pytest.fixture(scope="module")
def df():
    return synthetic_fx("EURUSD", years=1.2, seed=11)


@pytest.mark.parametrize("cut", [900, 1234, 3333, 5001, 6100, 7000])
def test_indicators_are_truncation_invariant(df, cut):
    p = StrategyParams()
    full = compute_indicators(df, p)
    part = compute_indicators(df.iloc[: cut + 1], p)
    assert_same(full, part, cut)          # includes the last row: decision uses closed data only
    lf, sf = make_signals(full, p, get_spec("EURUSD"))
    lp, sp_ = make_signals(part, p, get_spec("EURUSD"))
    assert (lf[: cut + 1] == lp).all() and (sf[: cut + 1] == sp_).all()


@pytest.mark.parametrize("cut", [1500, 4242, 6666])
def test_corrupting_the_future_changes_nothing_in_the_past(df, cut):
    p = StrategyParams()
    base = compute_indicators(df, p)
    poisoned = df.copy()
    fut = poisoned.index[cut + 1:]
    poisoned.loc[fut, ["open", "high", "low", "close"]] *= 1.37   # absurd future
    alt = compute_indicators(poisoned, p)
    assert_same(base, alt, cut)


def test_htf_trend_only_uses_closed_htf_bars(df):
    """At H1 bar i the H4 bar containing i+1 (still forming) must NOT influence trend[i]."""
    p = StrategyParams()
    base = compute_indicators(df, p)
    idx = np.where(df.index.hour == 9)[0]           # 09:00 bar: its H4 bar (08-12) is still forming
    i = int(idx[len(idx) // 2])
    poisoned = df.copy()
    poisoned.iloc[i + 1: i + 3, poisoned.columns.get_indexer(["open", "high", "low", "close"])] *= 3.0
    alt = compute_indicators(poisoned, p)
    assert_same(base, alt, i)


def test_engine_trades_identical_with_or_without_future_data():
    frames = {s: synthetic_fx(s, years=1.5, seed=5) for s in ("EURUSD", "USDJPY", "GBPUSD")}
    cfg = RunConfig(symbols=tuple(frames)).with_risk(dd_halt_cooldown_days=10)
    full = BacktestEngine(MarketData(frames), cfg).run()
    cutoff = frames["EURUSD"].index[int(len(frames["EURUSD"]) * 0.6)]
    trunc = {s: f[f.index <= cutoff] for s, f in frames.items()}
    part = BacktestEngine(MarketData(trunc), cfg).run()
    key = ["symbol", "side", "entry_time", "exit_time", "entry_price", "exit_price", "lots", "exit_reason"]
    a = full.trades[(full.trades.exit_time < cutoff)].reset_index(drop=True)
    b = part.trades[(part.trades.exit_time < cutoff) & (part.trades.exit_reason != "window_end")].reset_index(drop=True)
    assert len(a) > 20, "test needs trades to be meaningful"
    pd.testing.assert_frame_equal(a[key], b[key], check_exact=False, rtol=1e-12)
