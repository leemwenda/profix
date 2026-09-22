"""Fill / stop / cost behaviour on hand-crafted price paths (bid prices, EURUSD, 10-pip ATR).

Reference numbers: ATR=10 pips -> D=15 pips. Spread 1 pip. Long fill = open+spread.
Long:  entry 1.1001, SL 1.0986, TP 1.1046.   Short: entry 1.1000 (bid), SL 1.1015 (vs ASK), TP 1.0955.
"""
import pandas as pd
import pytest

from conftest import clean_cfg, run_scenario

pytestmark = pytest.mark.filterwarnings("ignore")


def only_trade(res):
    assert len(res.trades) == 1, res.trades
    return res.trades.iloc[0]


def test_long_entry_pays_spread_and_slippage():
    cfg = clean_cfg().with_costs(slippage_entry_pips=0.5)
    t = only_trade(run_scenario([(1.1000, 1.1004, 1.0996, 1.1000)] * 3, 1, cfg))
    assert t.entry_price == pytest.approx(1.1000 + 0.0001 + 0.00005)
    assert t.sl_initial == pytest.approx(t.entry_price - 0.0015)


def test_short_entry_at_bid_minus_slippage():
    cfg = clean_cfg().with_costs(slippage_entry_pips=0.5)
    t = only_trade(run_scenario([(1.1000, 1.1004, 1.0996, 1.1000)] * 3, -1, cfg))
    assert t.entry_price == pytest.approx(1.1000 - 0.00005)
    assert t.sl_initial == pytest.approx(t.entry_price + 0.0015)


def test_sl_and_tp_same_bar_resolves_to_stop_worst_case():
    t = only_trade(run_scenario([(1.1000, 1.1060, 1.0980, 1.1000)], 1))
    assert t.exit_reason == "sl"
    assert t.exit_price == pytest.approx(1.0986)
    assert -1.05 < t.r < -0.95


def test_take_profit_is_3R():
    t = only_trade(run_scenario([(1.1000, 1.1050, 1.0995, 1.1040)], 1))
    assert t.exit_reason == "tp"
    assert t.exit_price == pytest.approx(1.1046)
    assert t.r == pytest.approx(3.0, abs=0.02)


def test_gap_through_stop_fills_at_gap_price_plus_slippage():
    cfg = clean_cfg().with_costs(slippage_sl_pips=1.0)
    bars = [(1.1000, 1.1004, 1.0996, 1.1000),      # entry bar (normal)
            (1.0950, 1.0960, 1.0940, 1.0950)]      # next bar OPENS below the stop
    t = only_trade(run_scenario(bars, 1, cfg))
    assert t.exit_reason == "sl_gap"
    assert t.exit_price == pytest.approx(1.0950 - 0.0001)
    assert t.r < -2.0  # gap losses exceed 1R - the engine must not hide that


def test_long_stop_checked_on_bid_short_stop_on_ask():
    # bid low 1.0987 does NOT touch long SL 1.0986 ...
    res = run_scenario([(1.1000, 1.1004, 1.0987, 1.1000)] * 2 + [(1.1000, 1.1004, 1.0996, 1.1000)], 1)
    assert res.trades.iloc[0].exit_reason == "window_end"
    # ... but a bid high of 1.1014 + 1 pip spread = ask 1.1015 DOES touch the short SL
    t = only_trade(run_scenario([(1.1000, 1.1014, 1.0996, 1.1000)], -1))
    assert t.exit_reason == "sl"
    assert t.exit_price == pytest.approx(1.1015)


def test_break_even_then_stopped_at_entry_plus_offset():
    cfg = clean_cfg(be_r=1.0, be_offset_pips=1.0)
    bars = [(1.1000, 1.1020, 1.0995, 1.1018),   # touches +1.27R -> BE armed for NEXT bar
            (1.1018, 1.1019, 1.1000, 1.1005)]   # falls through BE
    t = only_trade(run_scenario(bars, 1, cfg))
    assert t.exit_reason == "sl"
    assert t.exit_price == pytest.approx(1.1001 + 0.0001)


def test_stop_change_does_not_apply_inside_the_bar_that_triggered_it():
    cfg = clean_cfg(be_r=1.0)
    # bar touches +1R and then dives through the ORIGINAL stop -> must be a full -1R loss
    t = only_trade(run_scenario([(1.1000, 1.1020, 1.0980, 1.0990)], 1, cfg))
    assert t.exit_reason == "sl" and t.r < -0.95


def test_trailing_stop_locks_profit_and_never_loosens():
    cfg = clean_cfg(be_r=1.0, trail_start_r=1.5, trail_atr_mult=2.0)
    bars = [(1.1000, 1.1030, 1.0995, 1.1028),
            (1.1028, 1.1029, 1.1005, 1.1010)]
    res = run_scenario(bars, 1, cfg)
    t = only_trade(res)
    assert t.exit_reason == "sl" and t.r > 0 and t.exit_price > 1.1005
    assert res.stop_loosen_attempts == 0


def test_time_exit_executes_at_next_open():
    cfg = clean_cfg(max_bars_in_trade=3)
    res = run_scenario([(1.1000, 1.1004, 1.0996, 1.1000)] * 6, 1, cfg)
    t = only_trade(res)
    assert t.exit_reason == "time_exit"
    assert t.exit_time - t.entry_time == pd.Timedelta(hours=3)


def test_weekend_flatten_on_friday():
    start = (pd.Timestamp("2024-01-19 18:00", tz="UTC") - pd.Timedelta(hours=400)).isoformat()
    cfg = clean_cfg(close_before_weekend=True, weekend_close_hour_utc=20)
    res = run_scenario([(1.1000, 1.1004, 1.0996, 1.1000)] * 5, 1, cfg, start=start)
    t = only_trade(res)
    assert t.exit_reason == "weekend_close"
    assert t.exit_time == pd.Timestamp("2024-01-19 20:00", tz="UTC")


def test_commission_is_included_in_sizing_and_charged():
    cfg = clean_cfg().with_costs(commission_per_lot_rt=7.0)
    t = only_trade(run_scenario([(1.1000, 1.1004, 1.0996, 1.1000)] * 2, 1, cfg))
    assert t.lots == pytest.approx(0.31)          # floor(50 / (150 + 7) = 0.318) -> 0.31
    assert t.commission == pytest.approx(7.0 * 0.31)
    assert t.risk_money <= 0.005 * 10_000 + 1e-6  # never exceeds the risk budget


def test_lot_below_broker_minimum_skips_trade_instead_of_rounding_up():
    cfg = clean_cfg().with_risk(risk_pct=0.005).with_costs()
    res = run_scenario([(1.1000, 1.1004, 1.0996, 1.1000)] * 2, 1, cfg)
    assert len(res.trades) == 0
    assert res.rejections.get("lot_below_min") == 1


def test_spread_gate_rejects_wide_spreads():
    cfg = clean_cfg().with_costs(spread_default_pips=4.0)   # > max_spread_pips 3.0
    res = run_scenario([(1.1000, 1.1004, 1.0996, 1.1000)] * 2, 1, cfg)
    assert len(res.trades) == 0 and res.rejections.get("spread_abs") == 1


def test_partial_close_books_profit_and_keeps_remainder():
    cfg = clean_cfg(partial_pct=0.5, partial_r=1.0, tp_r=6.0)
    bars = [(1.1000, 1.1020, 1.0995, 1.1018)] + [(1.1018, 1.1019, 1.1017, 1.1018)] * 2 + [(1.1018, 1.1019, 1.0900, 1.0950)]
    t = only_trade(run_scenario(bars, 1, cfg))
    assert t.lots == pytest.approx(0.33)
    assert t.gross != 0 and t.exit_reason in ("sl", "sl_gap")
