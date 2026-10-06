"""
MTF research adapter for the existing ProFX H1 backtest engine.

Architecture:

    H1 regime
        ↓
    M15 setup
        ↓
    M5 confirmation
        ↓
    M1 trigger
        ↓
    decision engine
        ↓
    H1-aligned signal_override
        ↓
    existing BacktestEngine

Important:
    - No engine.py changes.
    - Decisions are made only at H1 candle close.
    - Only candles whose CLOSE <= decision time are used.
    - Signals are stored on the H1 signal bar and therefore execute
      at the next H1 OPEN through BacktestEngine.
    - Native lower-timeframe gaps are never filled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import RunConfig
from .mtf import MTFData, TIMEFRAME_MINUTES
from .mtf_decision import build_decision_snapshot, validate_snapshot
from .mtf_features import MTFFeatures
from .mtf_regime import assess_regime
from .mtf_setup import assess_setup
from .mtf_decision_engine import (
    DecisionAssessment,
    MarketConditions,
    assess_decision,
)
from .specs import get_spec


@dataclass(frozen=True)
class MTFDecisionRecord:
    """
    Research record for one H1 decision point.
    """

    signal_time: pd.Timestamp
    decision_time: pd.Timestamp
    decision: DecisionAssessment


def _indicator_frame(
    frame: pd.DataFrame,
    *,
    atr_period: int = 14,
    ema_fast_period: int = 20,
    ema_slow_period: int = 50,
    momentum_period: int = 5,
) -> pd.DataFrame:
    """
    Precompute native-timeframe indicators once.

    This is deliberately equivalent to the calculations in
    mtf_features.calculate_bar_features(), but vectorized so a large
    historical backtest does not recalculate the entire series for
    every H1 decision.
    """

    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)

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

    candle_range = high - low
    body = (close - frame["open"].astype(float)).abs()

    body_ratio = body.div(
        candle_range.replace(0.0, np.nan)
    ).fillna(0.0)

    close_location = (
        (close - low)
        .div(candle_range.replace(0.0, np.nan))
        .fillna(0.5)
    )

    return pd.DataFrame(
        {
            "atr": atr,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "momentum": momentum,
            "candle_range": candle_range,
            "body": body,
            "body_ratio": body_ratio,
            "close_location": close_location,
        },
        index=frame.index,
    )


def _align_closed_rows(
    frame: pd.DataFrame,
    indicators: pd.DataFrame,
    decision_times: pd.DatetimeIndex,
    timeframe: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Align the latest CLOSED native candle to every decision timestamp.

    Returns:
        positions:
            Native dataframe positions, -1 when no closed bar exists.

        rows:
            Native OHLC + precomputed indicators aligned to decision times.

    No forward filling is performed.
    """

    duration = pd.Timedelta(
        minutes=TIMEFRAME_MINUTES[timeframe]
    )

    close_times = frame.index + duration

    positions = (
        close_times.searchsorted(
            decision_times,
            side="right",
        )
        - 1
    )

    valid = positions >= 0

    columns = [
        "open",
        "high",
        "low",
        "close",
        "spread",
    ]

    available_columns = [
        column
        for column in columns
        if column in frame.columns
    ]

    source = pd.concat(
        [
            frame[available_columns].astype(float),
            indicators,
        ],
        axis=1,
    )

    result = pd.DataFrame(
        index=decision_times,
        columns=source.columns,
        dtype=float,
    )

    if valid.any():
        result.loc[valid] = source.iloc[
            positions[valid]
        ].to_numpy()

    return positions, result


def _optional_float(value) -> Optional[float]:
    if pd.isna(value):
        return None

    return float(value)


def _features_from_row(
    timeframe: str,
    row: pd.Series,
) -> Optional[MTFFeatures]:
    """
    Convert an aligned native row into MTFFeatures.

    A completely unavailable row remains None.
    """

    if pd.isna(row.get("close", np.nan)):
        return None

    return MTFFeatures(
        timeframe=timeframe,
        close=float(row["close"]),
        atr=_optional_float(row.get("atr")),
        ema_fast=_optional_float(row.get("ema_fast")),
        ema_slow=_optional_float(row.get("ema_slow")),
        momentum=_optional_float(row.get("momentum")),
        candle_range=float(row["candle_range"]),
        body=float(row["body"]),
        body_ratio=float(row["body_ratio"]),
        close_location=float(row["close_location"]),
    )


def _volatility_score(
    atr: Optional[float],
    pip_size: float,
    cfg: RunConfig,
) -> tuple[int, bool]:
    """
    Score and validate H1 volatility.

    The XAUUSD config expresses ATR limits in engine pips.
    """

    if atr is None or not np.isfinite(atr) or atr <= 0:
        return 0, False

    atr_pips = atr / pip_size

    minimum = cfg.strategy.min_atr_pips
    maximum = cfg.strategy.max_atr_pips

    if minimum <= atr_pips <= maximum:
        return 5, True

    return 0, False


def _spread_score(
    spread: Optional[float],
    point: float,
    pip: float,
    cfg: RunConfig,
) -> tuple[int, bool]:
    """
    Score and validate the native MT5 spread.

    MT5 exported spread is normally expressed in points.
    """

    if spread is None or not np.isfinite(spread):
        return 0, False

    spread_price = float(spread) * point
    spread_pips = spread_price / pip

    if spread_pips <= cfg.risk.max_spread_pips:
        return 5, True

    return 0, False


def _session_score(
    decision_time: pd.Timestamp,
    cfg: RunConfig,
) -> tuple[int, bool]:
    """
    Use the same UTC session convention as the baseline strategy.

    End time is exclusive.
    """

    hour = (
        decision_time.hour
        + decision_time.minute / 60.0
    )

    start = cfg.strategy.session_start_utc
    end = cfg.strategy.session_end_utc

    allowed = start <= hour < end

    return (5, True) if allowed else (0, False)


def _conditions(
    h1_features: Optional[MTFFeatures],
    h1_spread: Optional[float],
    decision_time: pd.Timestamp,
    cfg: RunConfig,
    *,
    symbol: str,
) -> tuple[MarketConditions, bool]:
    """
    Build decision-layer market conditions and the hard market filter.
    """

    spec = get_spec(symbol)

    volatility_score, volatility_ok = _volatility_score(
        h1_features.atr if h1_features else None,
        spec.pip,
        cfg,
    )

    spread_score, spread_ok = _spread_score(
        h1_spread,
        spec.point,
        spec.pip,
        cfg,
    )

    session_score, session_ok = _session_score(
        decision_time,
        cfg,
    )

    conditions = MarketConditions(
        volatility_score=volatility_score,
        spread_score=spread_score,
        session_score=session_score,
    )

    market_ok = (
        volatility_ok
        and spread_ok
        and session_ok
    )

    return conditions, market_ok


def build_mtf_signals(
    mtf: MTFData,
    cfg: RunConfig,
    *,
    symbol: str = "XAUUSD",
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    list[MTFDecisionRecord],
]:
    """
    Build H1-aligned LONG/SHORT signal arrays for BacktestEngine.

    Signal semantics:

        signal[i] = decision made at the CLOSE of H1 bar i

    Existing BacktestEngine then queues that signal at the end of H1 bar i
    and executes it at H1 bar i+1 OPEN.

    Returns:
        signal_override:
            {
                symbol: (long_array, short_array)
            }

        records:
            detailed MTF decision records for diagnostics.
    """

    h1 = mtf.frame("H1")

    if h1.empty:
        raise ValueError("H1 data is empty")

    # Every H1 bar represents one decision point at its CLOSE.
    decision_times = (
        h1.index
        + pd.Timedelta(hours=1)
    )

    h1_indicators = _indicator_frame(
        h1,
        atr_period=14,
        ema_fast_period=20,
        ema_slow_period=50,
        momentum_period=5,
    )

    aligned: dict[str, pd.DataFrame] = {}
    positions: dict[str, np.ndarray] = {}

    for timeframe in (
        "H1",
        "M15",
        "M5",
        "M1",
    ):
        indicators = _indicator_frame(
            mtf.frame(timeframe),
            atr_period=14,
            ema_fast_period=20,
            ema_slow_period=50,
            momentum_period=5,
        )

        pos, rows = _align_closed_rows(
            mtf.frame(timeframe),
            indicators,
            decision_times,
            timeframe,
        )

        positions[timeframe] = pos
        aligned[timeframe] = rows

    long_signals = np.zeros(
        len(h1),
        dtype=bool,
    )

    short_signals = np.zeros(
        len(h1),
        dtype=bool,
    )

    records: list[MTFDecisionRecord] = []

    for i, decision_time in enumerate(decision_times):
        decision_time = pd.Timestamp(
            decision_time
        ).tz_convert("UTC")

        features: dict[
            str,
            Optional[MTFFeatures],
        ] = {}

        for timeframe in (
            "H1",
            "M15",
            "M5",
            "M1",
        ):
            features[timeframe] = _features_from_row(
                timeframe,
                aligned[timeframe].iloc[i],
            )

        # Hard snapshot validation against the actual native data.
        snapshot = build_decision_snapshot(
            mtf,
            decision_time,
        )
        validate_snapshot(snapshot)

        # Require the H1 bar itself to exist.
        if features["H1"] is None:
            continue

        regime = assess_regime(features)

        setup = assess_setup(
            regime.direction,
            features,
        )

        h1_row = aligned["H1"].iloc[i]
        h1_spread = _optional_float(
            h1_row.get("spread")
        )

        conditions, market_ok = _conditions(
            features["H1"],
            h1_spread,
            decision_time,
            cfg,
            symbol=symbol,
        )

        decision = assess_decision(
            regime,
            setup,
            conditions,
        )

        # The adapter applies the hard market filters because
        # signal_override bypasses the normal signal generator.
        executable = (
            decision.executable
            and market_ok
        )

        if executable:
            if decision.direction == "LONG":
                long_signals[i] = True

            elif decision.direction == "SHORT":
                short_signals[i] = True

        records.append(
            MTFDecisionRecord(
                signal_time=h1.index[i],
                decision_time=decision_time,
                decision=decision,
            )
        )

    return {
        symbol: (
            long_signals,
            short_signals,
        )
    }, records


def summarize_records(
    records: list[MTFDecisionRecord],
) -> dict[str, int]:
    """
    Compact diagnostic summary.
    """

    summary = {
        "decisions": len(records),
        "long": 0,
        "short": 0,
        "no_trade": 0,
        "premium": 0,
        "strong": 0,
        "normal": 0,
        "weak": 0,
    }

    for record in records:
        decision = record.decision

        if decision.direction == "LONG":
            summary["long"] += 1
        elif decision.direction == "SHORT":
            summary["short"] += 1
        else:
            summary["no_trade"] += 1

        if decision.tier == "PREMIUM":
            summary["premium"] += 1
        elif decision.tier == "STRONG":
            summary["strong"] += 1
        elif decision.tier == "NORMAL":
            summary["normal"] += 1
        elif decision.tier == "WEAK":
            summary["weak"] += 1

    return summary
