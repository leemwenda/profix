"""
Historical MTF research-window planning.

Determines which native timeframes are available over historical periods
without fabricating missing lower-timeframe data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .mtf import MTFData


@dataclass(frozen=True)
class ResearchWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    timeframes: tuple[str, ...]
    label: str


DEFAULT_LAYERS = ("H1", "M15", "M5", "M1")


def _coverage_start(mtf: MTFData, timeframe: str) -> pd.Timestamp:
    return mtf.frame(timeframe).index[0]


def _coverage_end(mtf: MTFData, timeframe: str) -> pd.Timestamp:
    return mtf.frame(timeframe).index[-1]


def build_research_windows(
    mtf: MTFData,
    *,
    required_timeframes: Iterable[str] = DEFAULT_LAYERS,
) -> list[ResearchWindow]:
    """
    Build maximal historical windows based on native dataset coverage.

    A timeframe becomes available only from its actual first native bar.
    No synthetic backfilling is performed.

    H1 is the base execution/regime layer.
    """

    required = tuple(required_timeframes)

    if "H1" not in required:
        raise ValueError("H1 must be included as the base timeframe")

    h1_start = _coverage_start(mtf, "H1")
    h1_end = _coverage_end(mtf, "H1")

    # Start/end dates where each lower timeframe becomes available.
    starts = {
        tf: _coverage_start(mtf, tf)
        for tf in required
    }

    boundaries = sorted(
        {
            h1_start,
            h1_end,
            *(
                starts[tf]
                for tf in required
                if tf != "H1"
                and h1_start < starts[tf] < h1_end
            ),
        }
    )

    windows: list[ResearchWindow] = []

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        available = tuple(
            tf
            for tf in required
            if starts[tf] <= left
            and _coverage_end(mtf, tf) >= right
        )

        if not available:
            continue

        if "M1" in available:
            label = "H1+M15+M5+M1"
        elif "M5" in available:
            label = "H1+M15+M5"
        elif "M15" in available:
            label = "H1+M15"
        else:
            label = "H1"

        windows.append(
            ResearchWindow(
                start=left,
                end=right,
                timeframes=available,
                label=label,
            )
        )

    # If H1 extends beyond the latest lower timeframe, add the tail.
    latest_lower_end = max(
        (
            _coverage_end(mtf, tf)
            for tf in required
            if tf != "H1"
        ),
        default=h1_start,
    )

    if latest_lower_end < h1_end:
        available = tuple(
            tf
            for tf in required
            if _coverage_start(mtf, tf) <= latest_lower_end
            and _coverage_end(mtf, tf) >= h1_end
        )

        # H1 is always available in this tail.
        if "H1" not in available:
            available = ("H1",)

        windows.append(
            ResearchWindow(
                start=latest_lower_end,
                end=h1_end,
                timeframes=available,
                label="H1",
            )
        )

    # Remove zero/negative windows and merge adjacent identical layers.
    cleaned: list[ResearchWindow] = []

    for window in windows:
        if window.start >= window.end:
            continue

        if (
            cleaned
            and cleaned[-1].end == window.start
            and cleaned[-1].timeframes == window.timeframes
        ):
            previous = cleaned[-1]

            cleaned[-1] = ResearchWindow(
                start=previous.start,
                end=window.end,
                timeframes=previous.timeframes,
                label=previous.label,
            )
        else:
            cleaned.append(window)

    return cleaned


def format_windows(windows: list[ResearchWindow]) -> str:
    lines = []

    for i, window in enumerate(windows, 1):
        lines.append(
            f"{i}. "
            f"{window.start} -> {window.end} | "
            f"{window.label} | "
            f"{window.end - window.start}"
        )

    return "\n".join(lines)
