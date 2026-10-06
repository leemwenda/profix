import numpy as np
import pandas as pd
import pytest

from profx.data import DataError
from profx.mtf import MTFData, validate_mtf


def make_frame(start: str, periods: int, freq: str) -> pd.DataFrame:
    idx = pd.date_range(
        start=start,
        periods=periods,
        freq=freq,
        tz="UTC",
    )

    close = np.arange(periods, dtype=float) + 100.0

    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "spread": np.ones(periods),
        },
        index=idx,
    )


@pytest.fixture
def mtf():
    frames = {
        "M1": make_frame("2026-01-01 00:00", 20, "1min"),
        "M5": make_frame("2026-01-01 00:00", 10, "5min"),
        "M15": make_frame("2026-01-01 00:00", 8, "15min"),
        "H1": make_frame("2026-01-01 00:00", 4, "1h"),
    }

    return MTFData(frames)


def test_mtf_validation(mtf):
    validate_mtf(mtf)


def test_common_window(mtf):
    start, end = mtf.common_window()

    assert start == pd.Timestamp("2026-01-01 00:00", tz="UTC")
    # M1 is the shortest dataset in this fixture, ending at 00:19.
    # The common window therefore ends at the earliest dataset boundary.
    assert end == pd.Timestamp("2026-01-01 00:19", tz="UTC")


def test_closed_bar_is_not_forming(mtf):
    # M5 bar 00:00 closes at 00:05.
    # At exactly 00:05 it is available.
    row = mtf.closed_bar_at_or_before(
        "M5",
        pd.Timestamp("2026-01-01 00:05", tz="UTC"),
    )

    assert row is not None
    assert row.name == pd.Timestamp("2026-01-01 00:00", tz="UTC")


def test_forming_bar_is_excluded(mtf):
    # At 00:02 the 00:00 M5 candle is still forming.
    row = mtf.closed_bar_at_or_before(
        "M5",
        pd.Timestamp("2026-01-01 00:02", tz="UTC"),
    )

    assert row is None


def test_alignment_uses_only_closed_bars(mtf):
    decisions = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-01 00:04", tz="UTC"),
            pd.Timestamp("2026-01-01 00:05", tz="UTC"),
            pd.Timestamp("2026-01-01 00:10", tz="UTC"),
        ]
    )

    aligned = mtf.align_closed("M5", decisions)

    # 00:04 -> no closed M5 candle.
    assert np.isnan(aligned.loc[decisions[0], "close"])

    # 00:05 -> 00:00 M5 candle.
    assert aligned.loc[decisions[1], "close"] == 100.0

    # 00:10 -> 00:05 M5 candle.
    assert aligned.loc[decisions[2], "close"] == 101.0


def test_future_bars_are_not_used(mtf):
    decision = pd.Timestamp("2026-01-01 00:10", tz="UTC")

    before = mtf.closed_bar_at_or_before("M5", decision)

    poisoned = mtf.frame("M5").copy()

    future = poisoned.index > pd.Timestamp(
        "2026-01-01 00:05",
        tz="UTC",
    )

    poisoned.loc[
        future,
        ["open", "high", "low", "close"],
    ] *= 100.0

    altered = MTFData(
        {
            "M1": mtf.frame("M1"),
            "M5": poisoned,
            "M15": mtf.frame("M15"),
            "H1": mtf.frame("H1"),
        }
    )

    after = altered.closed_bar_at_or_before("M5", decision)

    assert before["close"] == after["close"]


def test_gap_is_allowed(mtf):
    frame = mtf.frame("M5").copy()

    # Remove several candles to simulate a market closure.
    frame = frame.drop(frame.index[3:7])

    altered = MTFData(
        {
            "M1": mtf.frame("M1"),
            "M5": frame,
            "M15": mtf.frame("M15"),
            "H1": mtf.frame("H1"),
        }
    )

    validate_mtf(altered)


def test_sub_native_interval_is_rejected(mtf):
    frame = mtf.frame("M5").copy()

    # Force the third M5 bar to occur only one minute after
    # the second bar. This creates a sub-native interval.
    bad_index = frame.index.to_list()
    bad_index[2] = bad_index[1] + pd.Timedelta(minutes=1)

    frame.index = pd.DatetimeIndex(bad_index)

    # MTFData validates OHLC structure, while validate_mtf() performs
    # native-timeframe interval validation.
    altered = MTFData(
        {
            "M1": mtf.frame("M1"),
            "M5": frame,
            "M15": mtf.frame("M15"),
            "H1": mtf.frame("H1"),
        }
    )

    with pytest.raises(DataError):
        validate_mtf(altered)
