"""Position sizing, portfolio limits, circuit breakers, accounting identities."""
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from profx.config import ConfigError, RiskParams, RunConfig, StrategyParams, load_config
from profx.engine import BacktestEngine, MarketData
from profx.metrics import compute_metrics, drawdown_stats, max_consecutive, monthly_returns
from profx.specs import get_spec, round_lots_down

SYMS = ("EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD")


@pytest.fixture(scope="module")
def run(request):
    from profx.data import synthetic_fx
    frames = {s: synthetic_fx(s, years=3, seed=21) for s in SYMS}
    cfg = RunConfig(symbols=SYMS).with_risk(dd_halt_cooldown_days=20)
    md = MarketData(frames)
    return md, cfg, BacktestEngine(md, cfg).run()


# ---------------------------------------------------------------- sizing
def test_round_lots_never_up():
    spec = get_spec("EURUSD")
    assert round_lots_down(0.3599, spec) == 0.35
    assert round_lots_down(0.3, spec) == 0.30          # float artefact 0.3/0.01 = 29.999...
    assert round_lots_down(0.009, spec) == 0.0
    assert round_lots_down(float("nan"), spec) == 0.0
    assert round_lots_down(1e9, spec) == spec.vol_max


def test_realised_risk_of_stopped_trades_matches_budget_on_all_quote_currencies(run):
    """A trade stopped at its SL must lose ~1R whether the quote ccy is USD, JPY or CAD."""
    _, _, res = run
    t = res.trades
    sl = t[t.exit_reason == "sl"]
    assert len(sl) > 40
    for sym, g in sl.groupby("symbol"):
        assert -1.20 < g.r.median() < -0.95, (sym, g.r.median())


def test_risk_never_exceeds_budget(run):
    _, cfg, res = run
    cap = cfg.risk.risk_pct / 100 * res.equity.max()
    assert (res.trades.risk_money <= cap + 1e-6).all()
    assert (res.trades.lots >= 0.01).all()
    steps = res.trades.lots / 0.01
    assert np.allclose(steps, steps.round(), atol=1e-6)


# ---------------------------------------------------------------- portfolio limits
def _intervals(t):
    return [(r.entry_time, r.exit_time, r.symbol, r.side) for r in t.itertuples()]


def test_max_open_positions_and_one_per_symbol(run):
    _, cfg, res = run
    ev = []
    for a, b, s, _ in _intervals(res.trades):
        ev += [(a, 1, s), (b, -1, s)]
    ev.sort(key=lambda x: (x[0], x[1]))     # closes before opens at equal timestamps
    open_, per = 0, Counter()
    for _, d, s in ev:
        open_ += d
        per[s] += d
        assert open_ <= cfg.risk.max_open_positions
        assert per[s] <= cfg.risk.max_positions_per_symbol


def test_currency_direction_exposure_cap(run):
    _, cfg, res = run
    specs = {s: get_spec(s) for s in SYMS}
    for r in res.trades.itertuples():
        exp = Counter()
        for o in res.trades.itertuples():
            if o.Index != r.Index and o.entry_time <= r.entry_time < o.exit_time:
                sp = specs[o.symbol]
                s = 1 if o.side == "buy" else -1
                exp[(sp.base, s)] += 1
                exp[(sp.quote, -s)] += 1
        sp = specs[r.symbol]
        s = 1 if r.side == "buy" else -1
        assert exp[(sp.base, s)] + 1 <= cfg.risk.max_same_ccy_dir
        assert exp[(sp.quote, -s)] + 1 <= cfg.risk.max_same_ccy_dir


def test_cooldown_respected(run):
    _, cfg, res = run
    t = res.trades.sort_values("entry_time")
    for sym, g in t.groupby("symbol"):
        gap = g.entry_time.shift(-1) - g.exit_time
        assert (gap.dropna() >= pd.Timedelta(hours=cfg.risk.cooldown_bars)).all()


def test_daily_trade_cap(run):
    _, cfg, res = run
    per_day = res.trades.groupby(res.trades.entry_time.dt.date).size()
    assert per_day.max() <= cfg.risk.max_daily_trades


def test_stops_never_loosen_and_accounting_identity(run):
    _, cfg, res = run
    assert res.stop_loosen_attempts == 0
    assert res.equity.iloc[-1] == pytest.approx(cfg.initial_balance + res.trades.pnl.sum(), rel=1e-9)
    assert (res.trades.sl_initial > 0).all()


# ---------------------------------------------------------------- circuit breakers
def test_max_drawdown_breaker_halts_and_latches():
    from profx.data import synthetic_fx
    frames = {s: synthetic_fx(s, years=2, seed=8, trend_strength=0.0) for s in SYMS}
    cfg = RunConfig(symbols=SYMS).with_risk(max_drawdown_pct=3.0, dd_halt_cooldown_days=0)
    res = BacktestEngine(MarketData(frames), cfg).run()
    assert res.halted == "max_drawdown" and res.halted_at is not None
    assert (res.trades.entry_time <= res.halted_at).all(), "no NEW entries after the breaker trips"
    assert res.trades.exit_time.max() <= res.halted_at + pd.Timedelta(hours=2)   # flattened at next open
    assert res.signals.outcome.str.contains("halted").any()


def test_daily_loss_limit_blocks_and_flattens():
    from profx.data import synthetic_fx
    frames = {s: synthetic_fx(s, years=2, seed=9, trend_strength=0.0) for s in SYMS}
    cfg = RunConfig(symbols=SYMS).with_risk(daily_loss_limit_pct=0.4, dd_halt_cooldown_days=15)
    res = BacktestEngine(MarketData(frames), cfg).run()
    assert res.rejections.get("daily_loss_block", 0) > 0
    assert (res.trades.exit_reason == "daily_loss_limit").any()


# ---------------------------------------------------------------- config validation
def test_config_rejects_unsafe_values():
    with pytest.raises(ConfigError):
        RiskParams(risk_pct=5.0)
    with pytest.raises(ConfigError):
        RiskParams(risk_pct=1.0, max_total_risk_pct=0.5)
    with pytest.raises(ConfigError):
        StrategyParams(ema_fast=200, ema_slow=50)
    with pytest.raises(ConfigError):
        StrategyParams(min_clv=0.3)


def test_toml_unknown_keys_rejected(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("[risk]\nrisk_pctt = 1.0\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_default_toml_loads_and_equals_dataclass_defaults():
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[2] / "config" / "strategy.toml")
    assert cfg.strategy == StrategyParams() and cfg.risk == RiskParams()


# ---------------------------------------------------------------- metrics
def test_metric_helpers():
    assert max_consecutive(np.array([1, 1, 0, 1, 1, 1, 0])) == 3
    eq = pd.Series([100, 110, 99, 105, 120, 90.0])
    dd, pct = drawdown_stats(eq)
    assert dd == pytest.approx(30.0) and pct == pytest.approx(25.0)


def test_metrics_profit_factor_and_expectancy(run):
    _, _, res = run
    m = compute_metrics(res)
    t = res.trades
    assert m["total_trades"] == len(t)
    assert m["gross_profit"] == pytest.approx(t.pnl[t.pnl > 0].sum())
    assert m["gross_loss"] == pytest.approx(-t.pnl[t.pnl <= 0].sum())
    assert m["profit_factor"] == pytest.approx(m["gross_profit"] / m["gross_loss"])
    assert m["expectancy_money"] == pytest.approx(t.pnl.mean())
    assert not monthly_returns(res.equity, res.initial_balance).empty
