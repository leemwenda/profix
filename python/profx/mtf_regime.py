"""
Multi-timeframe directional regime analysis.

Higher timeframes establish directional permission.
Lower timeframes cannot override a strong higher-timeframe regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .mtf_features import MTFFeatures


@dataclass(frozen=True)
class RegimeAssessment:
    direction: str
    score: int
    h1_bias: str
    m15_bias: Optional[str]
    m5_bias: Optional[str]
    m1_bias: Optional[str]
    reasons: tuple[str, ...]

    @property
    def bullish(self) -> bool:
        return self.direction == "LONG"

    @property
    def bearish(self) -> bool:
        return self.direction == "SHORT"

    @property
    def neutral(self) -> bool:
        return self.direction == "NEUTRAL"


def _trend(
    features: Optional[MTFFeatures],
) -> str:
    if features is None:
        return "UNAVAILABLE"

    if (
        features.ema_fast is None
        or features.ema_slow is None
    ):
        return "NEUTRAL"

    if features.ema_fast > features.ema_slow:
        return "LONG"

    if features.ema_fast < features.ema_slow:
        return "SHORT"

    return "NEUTRAL"


def _signed_confirmation(
    h1: str,
    timeframe_bias: str,
    weight: int,
    conflict_penalty: int,
) -> int:
    """
    Return a signed contribution relative to the H1 direction.

    LONG H1:
        LONG confirmation  -> +weight
        SHORT conflict     -> -conflict_penalty

    SHORT H1:
        SHORT confirmation -> -weight
        LONG conflict      -> +conflict_penalty

    Neutral/unavailable lower timeframe contributes zero.
    """

    if h1 == "LONG":
        if timeframe_bias == "LONG":
            return weight

        if timeframe_bias == "SHORT":
            return -conflict_penalty

    elif h1 == "SHORT":
        if timeframe_bias == "SHORT":
            return -weight

        if timeframe_bias == "LONG":
            return conflict_penalty

    return 0


def assess_regime(
    features: dict[str, Optional[MTFFeatures]],
) -> RegimeAssessment:
    """
    Build a directional assessment.

    Weighting:
        H1  = primary directional regime
        M15 = confirmation
        M5  = confirmation
        M1  = trigger context only

    Maximum bullish score: +95
    Maximum bearish score: -95

    M1 cannot reverse an H1 regime.
    """

    h1 = _trend(features.get("H1"))
    m15 = _trend(features.get("M15"))
    m5 = _trend(features.get("M5"))
    m1 = _trend(features.get("M1"))

    reasons: list[str] = []

    if h1 == "UNAVAILABLE":
        return RegimeAssessment(
            direction="NEUTRAL",
            score=0,
            h1_bias=h1,
            m15_bias=m15,
            m5_bias=m5,
            m1_bias=m1,
            reasons=("H1 unavailable",),
        )

    score = 0

    # H1 establishes the main regime.
    if h1 == "LONG":
        score += 50
        reasons.append("H1 bullish")

    elif h1 == "SHORT":
        score -= 50
        reasons.append("H1 bearish")

    else:
        reasons.append("H1 neutral")

    # M15 confirmation.
    m15_contribution = _signed_confirmation(
        h1,
        m15,
        weight=25,
        conflict_penalty=15,
    )

    score += m15_contribution

    if m15 == h1:
        reasons.append("M15 confirms H1")
    elif m15 not in ("UNAVAILABLE", "NEUTRAL"):
        reasons.append("M15 conflicts with H1")

    # M5 confirmation.
    m5_contribution = _signed_confirmation(
        h1,
        m5,
        weight=15,
        conflict_penalty=10,
    )

    score += m5_contribution

    if m5 == h1:
        reasons.append("M5 confirms H1")
    elif m5 not in ("UNAVAILABLE", "NEUTRAL"):
        reasons.append("M5 conflicts with H1")

    # M1 is deliberately weak.
    m1_contribution = _signed_confirmation(
        h1,
        m1,
        weight=5,
        conflict_penalty=5,
    )

    score += m1_contribution

    if m1 == h1:
        reasons.append("M1 confirms H1")
    elif m1 not in ("UNAVAILABLE", "NEUTRAL"):
        reasons.append("M1 conflicts with H1")

    if score >= 50:
        direction = "LONG"
    elif score <= -50:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    return RegimeAssessment(
        direction=direction,
        score=max(-100, min(100, score)),
        h1_bias=h1,
        m15_bias=m15,
        m5_bias=m5,
        m1_bias=m1,
        reasons=tuple(reasons),
    )
