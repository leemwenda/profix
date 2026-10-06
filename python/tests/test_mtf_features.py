import pandas as pd

from profx.mtf import MTFData
from profx.mtf_decision import build_decision_snapshot
from profx.mtf_features import (
    calculate_bar_features,
    calculate_snapshot_features,
)


def frame(start, end, freq):
    idx = pd.date_range(
        start=start,
        end=end,
        freq=freq,
        tz="UTC",
    )

    values = range(1000, 1000 + len(idx))

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
                "2026-01-01",
                "2026-10-06",
                "1h",
            ),
            "M15": frame(
                "2026-01-01",
                "2026-10-06",
                "15min",
            ),
            "M5": frame(
                "2026-01-01",
                "2026-10-06",
                "5min",
            ),
            "M1": frame(
                "2026-01-01",
                "2026-10-06",
                "1min",
            ),
        }
    )


def test_features_use_closed_bar_only():
    mtf = make_mtf()

    snapshot = build_decision_snapshot(
        mtf,
        "2026-08-14 16:28:30+00:00",
    )

    bar = snapshot.bar("M5")

    features = calculate_bar_features(
        mtf.frames["M5"],
        bar,
    )

    assert features.timeframe == "M5"
    assert features.close == bar.close
    assert features.candle_range >= 0
    assert 0.0 <= features.body_ratio <= 1.0
    assert 0.0 <= features.close_location <= 1.0


def test_indicators_are_none_until_warmup():
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
        "2026-01-01 05:30:00+00:00",
    )

    bar = snapshot.bar("H1")

    features = calculate_bar_features(
        mtf.frames["H1"],
        bar,
    )

    assert features.atr is None
    assert features.ema_slow is None


def test_snapshot_features_respects_available_timeframes():
    mtf = make_mtf()

    snapshot = build_decision_snapshot(
        mtf,
        "2026-08-14 16:28:30+00:00",
    )

    features = calculate_snapshot_features(
        mtf.frames,
        snapshot,
    )

    assert set(features) == {"H1", "M15", "M5", "M1"}

    for timeframe in features:
        assert features[timeframe] is not None
        assert features[timeframe].timeframe == timeframe
