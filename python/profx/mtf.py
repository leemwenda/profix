"""
Multi-timeframe market data layer for ProFX.

Purpose:
    Load independently exported MT5 OHLC data for:
        H1 -> regime / structure
        M15 -> setup
        M5 -> confirmation
        M1 -> entry

Important:
    - No synthetic filling of market gaps.
    - No forward-looking values.
    - Native timeframe data remains independent.
    - A timeframe may only contribute information from bars that are
      CLOSED at the decision timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .data import DataError, load_csv, validate_ohlc


TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "H1": 60,
}


@dataclass(frozen=True)
class MTFFrame:
    """One native timeframe."""

    timeframe: str
    frame: pd.DataFrame

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.frame.index

    @property
    def first(self) -> pd.Timestamp:
        return self.frame.index[0]

    @property
    def last(self) -> pd.Timestamp:
        return self.frame.index[-1]

    @property
    def rows(self) -> int:
        return len(self.frame)


@dataclass(frozen=True)
class MTFData:
    """
    Collection of independently loaded native timeframe datasets.

    Frames are keyed by:
        M1, M5, M15, H1
    """

    frames: Mapping[str, pd.DataFrame]

    def __post_init__(self) -> None:
        required = {"M1", "M5", "M15", "H1"}
        missing = required - set(self.frames)
        if missing:
            raise DataError(f"Missing MTF frames: {sorted(missing)}")

        for tf, frame in self.frames.items():
            if tf not in TIMEFRAME_MINUTES:
                raise DataError(f"Unsupported timeframe: {tf}")

            validate_ohlc(frame, f"XAUUSD_{tf}")

    def frame(self, timeframe: str) -> pd.DataFrame:
        try:
            return self.frames[timeframe]
        except KeyError as exc:
            raise DataError(f"MTF timeframe not loaded: {timeframe}") from exc

    def coverage(self) -> dict[str, dict[str, object]]:
        """Return compact coverage information for diagnostics."""

        return {
            tf: {
                "rows": len(df),
                "first": df.index[0],
                "last": df.index[-1],
            }
            for tf, df in self.frames.items()
        }

    def common_start(self) -> pd.Timestamp:
        """Latest first timestamp shared by all loaded timeframes."""

        return max(df.index[0] for df in self.frames.values())

    def common_end(self) -> pd.Timestamp:
        """Earliest last timestamp shared by all loaded timeframes."""

        return min(df.index[-1] for df in self.frames.values())

    def common_window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        start = self.common_start()
        end = self.common_end()

        if start >= end:
            raise DataError(
                f"No common MTF window: start={start}, end={end}"
            )

        return start, end

    def closed_bar_at_or_before(
        self,
        timeframe: str,
        decision_time: pd.Timestamp,
    ) -> pd.Series | None:
        """
        Return the latest bar whose OPEN time is strictly before decision_time.

        A bar opening exactly at decision_time is still forming and therefore
        cannot be used.

        Example:
            M5 bar 12:00-12:05
            decision at 12:05
            -> 12:00 bar is available.

            decision at 12:02
            -> 12:00 bar is NOT considered closed yet.
        """

        df = self.frame(timeframe)

        if decision_time.tzinfo is None:
            raise DataError("decision_time must be timezone-aware UTC")

        decision_time = decision_time.tz_convert("UTC")

        # Native bars are represented by OPEN timestamps.
        # For a timeframe of N minutes, a bar opened at T closes at:
        #     T + N minutes
        #
        # Therefore:
        #     bar_open + duration <= decision_time
        #     bar_open <= decision_time - duration
        duration = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
        cutoff = decision_time - duration

        pos = df.index.searchsorted(cutoff, side="right") - 1

        if pos < 0:
            return None

        return df.iloc[pos]

    def closed_bars_at_or_before(
        self,
        timeframe: str,
        decision_time: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return all fully closed native bars available by decision_time."""

        df = self.frame(timeframe)

        if decision_time.tzinfo is None:
            raise DataError("decision_time must be timezone-aware UTC")

        decision_time = decision_time.tz_convert("UTC")
        duration = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
        cutoff = decision_time - duration

        pos = df.index.searchsorted(cutoff, side="right")

        return df.iloc[:pos]

    def align_closed(
        self,
        timeframe: str,
        decision_times: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Align the latest CLOSED native bar to each decision timestamp.

        This uses only historical bars whose CLOSE time is <= decision time.

        Missing historical coverage remains NaN; it is never forward-filled
        across unavailable history.
        """

        df = self.frame(timeframe)

        if not isinstance(decision_times, pd.DatetimeIndex):
            decision_times = pd.DatetimeIndex(decision_times)

        if decision_times.tz is None:
            decision_times = decision_times.tz_localize("UTC")
        else:
            decision_times = decision_times.tz_convert("UTC")

        duration = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
        close_times = df.index + duration

        # For every decision time, select the latest bar whose CLOSE is <=
        # that decision time.
        positions = close_times.searchsorted(
            decision_times,
            side="right",
        ) - 1

        valid = positions >= 0

        result = pd.DataFrame(
            index=decision_times,
            columns=df.columns,
            dtype=float,
        )

        if valid.any():
            result.loc[valid] = df.iloc[positions[valid]].to_numpy()

        return result


def load_mtf(
    data_dir: str | Path,
    symbol: str = "XAUUSD",
    server_utc_offset_hours: float | None = None,
) -> MTFData:
    """
    Load native MT5 exports:

        <data_dir>/<symbol>_M1.csv
        <data_dir>/<symbol>_M5.csv
        <data_dir>/<symbol>_M15.csv
        <data_dir>/<symbol>_H1.csv
    """

    data_dir = Path(data_dir)

    frames: dict[str, pd.DataFrame] = {}

    for tf in ("M1", "M5", "M15", "H1"):
        path = data_dir / f"{symbol}_{tf}.csv"

        if not path.exists():
            raise DataError(f"Missing MTF data file: {path}")

        frame = load_csv(
            path,
            server_utc_offset_hours=server_utc_offset_hours,
        )

        validate_ohlc(frame, f"{symbol}_{tf}")

        frames[tf] = frame

    return MTFData(frames)


def validate_native_timeframe(
    frame: pd.DataFrame,
    timeframe: str,
) -> None:
    """
    Validate that timestamps are consistent with the native timeframe.

    We deliberately do NOT require continuous timestamps because broker
    sessions, weekends and holidays create legitimate gaps.
    """

    if timeframe not in TIMEFRAME_MINUTES:
        raise DataError(f"Unsupported timeframe: {timeframe}")

    validate_ohlc(frame, f"native {timeframe}")

    if len(frame) < 2:
        return

    expected = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    deltas = frame.index.to_series().diff().dropna()

    # A timestamp occurring earlier than the expected interval is invalid.
    # Larger gaps are allowed because markets close.
    bad = deltas[deltas < expected]

    if not bad.empty:
        raise DataError(
            f"{timeframe}: found {len(bad)} timestamp intervals "
            f"shorter than the native {expected} interval"
        )


def validate_mtf(mtf: MTFData) -> None:
    """Run structural validation across all native frames."""

    for tf, frame in mtf.frames.items():
        validate_native_timeframe(frame, tf)

    start, end = mtf.common_window()

    if not start < end:
        raise DataError(
            f"Invalid common MTF window: {start} -> {end}"
        )
