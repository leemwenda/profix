import pandas as pd

from profx.mtf import MTFData
from profx.mtf_decision import (
    build_decision_snapshot,
    validate_snapshot,
)


def frame(start, end, freq, base=1000):
    idx = pd.date_range(
        start=start,
        end=end,
        freq=freq,
        tz="UTC",
    )

    values = range(base, base + len(idx))

    return pd.DataFrame(
        {
            "open": values,
            "high": [x + 2 for x in values],
            "low": [x - 1 for x in values],
            "close": [x + 1 for x in values],
            "spread": [5] * len(idx),
        },
        index=idx,
    )


def make_mtf():
    return MTFData(
        {
            "H1": frame(
                "2026-06-22 00:00",
                "2026-10-06 18:00",
                "1h",
            ),
            "M15": frame(
                "2026-06-22 00:00",
                "2026-10-06 18:00",
                "15min",
            ),
            "M5": frame(
                "2026-06-22 00:00",
                "2026-10-06 18:00",
                "5min",
            ),
            "M1": frame(
                "2026-06-22 00:00",
                "2026-10-06 18:00",
                "1min",
            ),
        }
    )


def test_only_closed_bars_are_returned():
    mtf = make_mtf()

    snapshot = build_decision_snapshot(
        mtf,
        "2026-08-14 16:28:30+00:00",
    )

    assert snapshot.bar("H1").bar_time == pd.Timestamp(
        "2026-08-14 15:00:00+00:00"
    )

    assert snapshot.bar("M15").bar_time == pd.Timestamp(
        "2026-08-14 16:00:00+00:00"
    )

    assert snapshot.bar("M5").bar_time == pd.Timestamp(
        "2026-08-14 16:20:00+00:00"
    )

    assert snapshot.bar("M1").bar_time == pd.Timestamp(
        "2026-08-14 16:27:00+00:00"
    )

    validate_snapshot(snapshot)


def test_current_forming_bar_is_never_used():
    mtf = make_mtf()

    snapshot = build_decision_snapshot(
        mtf,
        "2026-08-14 16:27:30+00:00",
    )

    # 16:27 M1 candle closes at 16:28, so it is NOT available.
    assert snapshot.bar("M1") is not None
    assert snapshot.bar("M1").bar_time == pd.Timestamp(
        "2026-08-14 16:26:00+00:00"
    )

    validate_snapshot(snapshot)


def test_exact_close_is_allowed():
    mtf = make_mtf()

    snapshot = build_decision_snapshot(
        mtf,
        "2026-08-14 16:28:00+00:00",
    )

    # The 16:27 M1 candle closes exactly at 16:28.
    assert snapshot.bar("M1").bar_time == pd.Timestamp(
        "2026-08-14 16:27:00+00:00"
    )

    validate_snapshot(snapshot)


def test_missing_early_timeframe_returns_none():
    mtf = MTFData(
        {
            "H1": frame(
                "2026-01-01",
                "2026-01-02",
                "1h",
            ),
            "M15": frame(
                "2026-01-01",
                "2026-01-02",
                "15min",
            ),
            "M5": frame(
                "2026-01-01",
                "2026-01-02",
                "5min",
            ),
            "M1": frame(
                "2026-01-01",
                "2026-01-02",
                "1min",
            ),
        }
    )

    snapshot = build_decision_snapshot(
        mtf,
        "2025-12-01 12:00:00+00:00",
    )

    assert snapshot.bar("H1") is None
    assert snapshot.bar("M15") is None
    assert snapshot.bar("M5") is None
    assert snapshot.bar("M1") is None
