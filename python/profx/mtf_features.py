"""
Multi-timeframe feature calculations.

All features are calculated from native candles.
A feature at decision time may only use candles that have
already closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .mtf_decision import DecisionSnapshot, TimeframeBar


@dataclass(frozen=True)
class MTFFeatures:
    timeframe: str
    close: float
    atr: Optional[float]
    ema_fast: Optional[float]
    ema_slow: Optional[float]
    momentum: Optional[float]
    candle_range: float
    body: float
    body_ratio: float
    close_location: float


def _safe_float(value) -> Optional[float]:
    if pd.isna(value):
        return None
    return float(value)


def calculate_bar_features(
    frame: pd.DataFrame,
    bar: TimeframeBar,
    *,
    atr_period: int = 14,
    ema_fast_period: int = 20,
    ema_slow_period: int = 50,
    momentum_period: int = 5,
) -> MTFFeatures:
    """
    Calculate features using data through the selected CLOSED bar only.
    """

    data = frame.loc[:bar.bar_time].copy()

    high = data["high"].astype(float)
    low = data["low"].astype(float)
    close = data["close"].astype(float)

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(
        atr_period,
        min_periods=atr_period,
    ).mean()

    ema_fast = close.ewm(
        span=ema_fast_period,
        adjust=False,
        min_periods=ema_fast_period,
    ).mean()

    ema_slow = close.ewm(
        span=ema_slow_period,
        adjust=False,
        min_periods=ema_slow_period,
    ).mean()

    momentum = close.pct_change(momentum_period)

    candle_range = bar.high - bar.low
    body = abs(bar.close - bar.open)

    if candle_range > 0:
        body_ratio = body / candle_range
        close_location = (
            (bar.close - bar.low) / candle_range
        )
    else:
        body_ratio = 0.0
        close_location = 0.5

    return MTFFeatures(
        timeframe=bar.timeframe,
        close=bar.close,
        atr=_safe_float(atr.iloc[-1]),
        ema_fast=_safe_float(ema_fast.iloc[-1]),
        ema_slow=_safe_float(ema_slow.iloc[-1]),
        momentum=_safe_float(momentum.iloc[-1]),
        candle_range=candle_range,
        body=body,
        body_ratio=body_ratio,
        close_location=close_location,
    )


def calculate_snapshot_features(
    frames: dict[str, pd.DataFrame],
    snapshot: DecisionSnapshot,
    *,
    atr_period: int = 14,
    ema_fast_period: int = 20,
    ema_slow_period: int = 50,
    momentum_period: int = 5,
) -> dict[str, Optional[MTFFeatures]]:
    """
    Calculate features for every available timeframe in a snapshot.
    """

    result: dict[str, Optional[MTFFeatures]] = {}

    for timeframe, bar in snapshot.bars.items():
        if bar is None:
            result[timeframe] = None
            continue

        result[timeframe] = calculate_bar_features(
            frames[timeframe],
            bar,
            atr_period=atr_period,
            ema_fast_period=ema_fast_period,
            ema_slow_period=ema_slow_period,
            momentum_period=momentum_period,
        )

    return result
