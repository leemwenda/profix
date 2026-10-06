import numpy as np
import pandas as pd

from profx.config import RunConfig, load_config
from profx.mtf import MTFData
from profx.mtf_backtest import (
    _align_closed_rows,
    _indicator_frame,
    _session_score,
    _spread_score,
    _volatility_score,
    build_mtf_signals,
    summarize_records,
)
from profx.mtf_decision import build_decision_snapshot


def frame(start, end, freq, base=1000, spread=5):
    idx = pd.date_range(
        start=start,
        end=end,
        freq=freq,
        tz="UTC",
    )

    values = np.arange(base, base + len(idx), dtype=float)

    return pd.DataFrame(
        {
            "open": values,
            "high": values + 2.0,
            "low": values - 1.0,
            "close": values + 1.0,
            "spread": np.full(len(idx), spread, dtype=float),
        },
        index=idx,
    )


def make_mtf(
    *,
    h1_start="2026-06-22 00:00",
    h1_end="2026-06-24 23:00",
    m15_start="2026-06-22 00:00",
    m15_end="2026-06-24 23:45",
    m5_start="2026-06-22 00:00",
    m5_end="2026-06-24 23:55",
    m1_start="2026-06-22 00:00",
    m1_end="2026-06-24 23:59",
):
    return MTFData(
        {
            "H1": frame(h1_start, h1_end, "1h", base=1000),
            "M15": frame(m15_start, m15_end, "15min", base=2000),
            "M5": frame(m5_start, m5_end, "5min", base=3000),
            "M1": frame(m1_start, m1_end, "1min", base=4000),
        }
    )


def test_alignment_uses_only_fully_closed_native_bars():
    mtf = make_mtf()

    decision_times = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-06-23 10:00:00+00:00"),
            pd.Timestamp("2026-06-23 10:00:01+00:00"),
        ]
    )

    indicators = _indicator_frame(mtf.frame("M5"))

    positions, rows = _align_closed_rows(
        mtf.frame("M5"),
        indicators,
        decision_times,
        "M5",
    )

    # At exactly 10:00, the 09:55 M5 bar has just closed.
    assert rows.iloc[0]["close"] == mtf.frame("M5").loc[
        "2026-06-23 09:55:00+00:00",
        "close",
    ]

    # One second later the same closed candle remains selected.
    assert positions[0] == positions[1]
    assert rows.iloc[0]["close"] == rows.iloc[1]["close"]


def test_exact_close_boundary_matches_snapshot():
    mtf = make_mtf()

    decision_time = pd.Timestamp(
        "2026-06-23 10:00:00+00:00"
    )

    snapshot = build_decision_snapshot(
        mtf,
        decision_time,
    )

    indicators = _indicator_frame(mtf.frame("M5"))

    positions, rows = _align_closed_rows(
        mtf.frame("M5"),
        indicators,
        pd.DatetimeIndex([decision_time]),
        "M5",
    )

    aligned_position = positions[0]
    snapshot_bar = snapshot.bar("M5")

    assert aligned_position >= 0
    assert snapshot_bar is not None
    assert (
        mtf.frame("M5").index[aligned_position]
        == snapshot_bar.bar_time
    )


def test_missing_lower_timeframe_history_is_not_filled():
    mtf = make_mtf(
        m5_start="2026-06-23 12:00",
        m5_end="2026-06-24 23:55",
        m1_start="2026-06-23 12:00",
        m1_end="2026-06-24 23:59",
    )

    decision_time = pd.Timestamp(
        "2026-06-23 10:00:00+00:00"
    )

    for timeframe in ("M5", "M1"):
        indicators = _indicator_frame(
            mtf.frame(timeframe)
        )

        positions, rows = _align_closed_rows(
            mtf.frame(timeframe),
            indicators,
            pd.DatetimeIndex([decision_time]),
            timeframe,
        )

        assert positions[0] == -1
        assert pd.isna(rows.iloc[0]["close"])


def test_signal_arrays_match_h1_length():
    mtf = make_mtf()
    cfg = RunConfig()

    signals, records = build_mtf_signals(
        mtf,
        cfg,
    )

    long_signals, short_signals = signals["XAUUSD"]

    assert len(long_signals) == len(mtf.frame("H1"))
    assert len(short_signals) == len(mtf.frame("H1"))
    assert len(records) <= len(mtf.frame("H1"))


def test_signal_arrays_are_boolean():
    mtf = make_mtf()
    cfg = RunConfig()

    signals, _ = build_mtf_signals(
        mtf,
        cfg,
    )

    long_signals, short_signals = signals["XAUUSD"]

    assert long_signals.dtype == np.bool_
    assert short_signals.dtype == np.bool_
    assert not np.any(long_signals & short_signals)


def test_repeated_build_is_deterministic():
    mtf = make_mtf()
    cfg = RunConfig()

    first_signals, first_records = build_mtf_signals(
        mtf,
        cfg,
    )

    second_signals, second_records = build_mtf_signals(
        mtf,
        cfg,
    )

    first_long, first_short = first_signals["XAUUSD"]
    second_long, second_short = second_signals["XAUUSD"]

    np.testing.assert_array_equal(
        first_long,
        second_long,
    )
    np.testing.assert_array_equal(
        first_short,
        second_short,
    )

    assert summarize_records(first_records) == (
        summarize_records(second_records)
    )


def test_session_filter_uses_utc_and_excludes_end():
    cfg = RunConfig()

    score, ok = _session_score(
        pd.Timestamp("2026-06-23 07:00:00+00:00"),
        cfg,
    )
    assert score == 5
    assert ok

    score, ok = _session_score(
        pd.Timestamp("2026-06-23 19:00:00+00:00"),
        cfg,
    )
    assert score == 0
    assert not ok


def test_spread_filter_rejects_excessive_spread():
    cfg = load_config("config/xauusd.toml")

    score, ok = _spread_score(
        spread=20,
        point=0.01,
        pip=0.01,
        cfg=cfg,
    )

    assert score == 5
    assert ok

    score, ok = _spread_score(
        spread=31,
        point=0.01,
        pip=0.01,
        cfg=cfg,
    )

    assert score == 0
    assert not ok


def test_volatility_filter_rejects_missing_atr():
    cfg = RunConfig()

    score, ok = _volatility_score(
        atr=None,
        pip_size=0.01,
        cfg=cfg,
    )

    assert score == 0
    assert not ok
