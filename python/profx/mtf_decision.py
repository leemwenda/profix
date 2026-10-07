"""
Leak-free multi-timeframe decision snapshots.

At a decision timestamp, only COMPLETED native candles are available.

Example:
    decision = 16:28:30

    H1  -> 15:00 bar
    M15 -> 16:00 bar
    M5  -> 16:20 bar
    M1  -> 16:27 bar

The currently forming candle is never used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .mtf import MTFData


@dataclass(frozen=True)
class TimeframeBar:
    timeframe: str
    bar_time: pd.Timestamp
    close_time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    spread: Optional[float]


@dataclass(frozen=True)
class DecisionSnapshot:
    decision_time: pd.Timestamp
    bars: dict[str, Optional[TimeframeBar]]

    def available(self, timeframe: str) -> bool:
        return self.bars.get(timeframe) is not None

    def bar(self, timeframe: str) -> Optional[TimeframeBar]:
        return self.bars.get(timeframe)


def _to_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts


def _get_closed_bar(
    mtf: MTFData,
    timeframe: str,
    decision_time: pd.Timestamp,
) -> Optional[TimeframeBar]:
    frame = mtf.frame(timeframe)

    from .mtf import TIMEFRAME_MINUTES
    duration = TIMEFRAME_MINUTES[timeframe]

    cutoff = decision_time - pd.Timedelta(minutes=duration)

    pos = frame.index.searchsorted(cutoff, side="right") - 1

    if pos < 0:
        return None

    row = frame.iloc[pos]
    bar_time = frame.index[pos]
    close_time = bar_time + pd.Timedelta(minutes=duration)

    spread = None
    if "spread" in frame.columns:
        value = row["spread"]
        if pd.notna(value):
            spread = float(value)

    return TimeframeBar(
        timeframe=timeframe,
        bar_time=bar_time,
        close_time=close_time,
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        spread=spread,
    )


def build_decision_snapshot(
    mtf: MTFData,
    decision_time,
    *,
    timeframes=("H1", "M15", "M5", "M1"),
) -> DecisionSnapshot:
    """
    Build a leak-free snapshot of all requested native timeframes.

    A bar is usable only if its CLOSE time is <= decision_time.
    """

    decision_time = _to_timestamp(decision_time)

    bars = {
        timeframe: _get_closed_bar(
            mtf,
            timeframe,
            decision_time,
        )
        for timeframe in timeframes
    }

    return DecisionSnapshot(
        decision_time=decision_time,
        bars=bars,
    )


def validate_snapshot(snapshot: DecisionSnapshot) -> None:
    """
    Hard anti-lookahead validation.

    Every returned bar must have closed by the decision timestamp.
    """

    for timeframe, bar in snapshot.bars.items():
        if bar is None:
            continue

        if bar.close_time > snapshot.decision_time:
            raise AssertionError(
                f"{timeframe} lookahead detected: "
                f"close={bar.close_time}, "
                f"decision={snapshot.decision_time}"
            )
