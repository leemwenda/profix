from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .mtf_decision import DecisionSnapshot
from .mtf_features import MTFFeatures


@dataclass(frozen=True)
class SetupAssessment:
    direction: str
    score: int
    m15_valid: bool
    m5_valid: bool
    m1_valid: bool
    reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        # M15 is the primary setup timeframe.
        # Lower timeframes can confirm the setup but cannot create
        # a valid setup when M15 structure is unavailable.
        return (
            self.direction in {"LONG", "SHORT"}
            and self.m15_valid
            and self.score >= 15
        )

    @property
    def bullish(self) -> bool:
        return self.direction == "LONG"

    @property
    def bearish(self) -> bool:
        return self.direction == "SHORT"


def _direction_from_regime(regime_direction: str) -> str:
    if regime_direction in {"LONG", "SHORT"}:
        return regime_direction
    return "NEUTRAL"


def _m15_setup(
    direction: str,
    features: Optional[MTFFeatures],
) -> tuple[bool, str]:
    if features is None:
        return False, "M15 unavailable"

    if features.atr is None or features.ema_fast is None or features.ema_slow is None:
        return False, "M15 indicators not warmed up"

    if direction == "LONG":
        if features.close > features.ema_fast and features.ema_fast >= features.ema_slow:
            return True, "M15 bullish structure"
        return False, "M15 structure not bullish"

    if direction == "SHORT":
        if features.close < features.ema_fast and features.ema_fast <= features.ema_slow:
            return True, "M15 bearish structure"
        return False, "M15 structure not bearish"

    return False, "No directional regime"


def _m5_confirmation(
    direction: str,
    features: Optional[MTFFeatures],
) -> tuple[bool, str]:
    if features is None:
        return False, "M5 unavailable"

    if features.atr is None or features.momentum is None:
        return False, "M5 indicators not warmed up"

    if direction == "LONG":
        if features.momentum > 0 and features.close > features.ema_fast:
            return True, "M5 momentum confirms LONG"
        return False, "M5 momentum does not confirm LONG"

    if direction == "SHORT":
        if features.momentum < 0 and features.close < features.ema_fast:
            return True, "M5 momentum confirms SHORT"
        return False, "M5 momentum does not confirm SHORT"

    return False, "No directional regime"


def _m1_trigger(
    direction: str,
    features: Optional[MTFFeatures],
) -> tuple[bool, str]:
    if features is None:
        return False, "M1 unavailable"

    if features.atr is None or features.momentum is None:
        return False, "M1 indicators not warmed up"

    if direction == "LONG":
        if (
            features.momentum > 0
            and features.close_location >= 0.60
            and features.close > features.ema_fast
        ):
            return True, "M1 bullish trigger"

        return False, "M1 trigger not bullish"

    if direction == "SHORT":
        if (
            features.momentum < 0
            and features.close_location <= 0.40
            and features.close < features.ema_fast
        ):
            return True, "M1 bearish trigger"

        return False, "M1 trigger not bearish"

    return False, "No directional regime"


def assess_setup(
    regime_direction: str,
    features: dict[str, Optional[MTFFeatures]],
) -> SetupAssessment:
    """
    Evaluate the setup after the higher-timeframe regime has been established.

    Scoring:
        M15 structure = 15
        M5 confirmation = 10
        M1 trigger     = 5

    Maximum = 30.

    H1/regime remains the directional authority. This function never
    creates a direction independently of the supplied regime.
    """
    direction = _direction_from_regime(regime_direction)

    if direction == "NEUTRAL":
        return SetupAssessment(
            direction="NEUTRAL",
            score=0,
            m15_valid=False,
            m5_valid=False,
            m1_valid=False,
            reasons=("Regime is neutral",),
        )

    m15_ok, m15_reason = _m15_setup(direction, features.get("M15"))
    m5_ok, m5_reason = _m5_confirmation(direction, features.get("M5"))
    m1_ok, m1_reason = _m1_trigger(direction, features.get("M1"))

    score = (
        15 if m15_ok else 0
    ) + (
        10 if m5_ok else 0
    ) + (
        5 if m1_ok else 0
    )

    return SetupAssessment(
        direction=direction,
        score=score,
        m15_valid=m15_ok,
        m5_valid=m5_ok,
        m1_valid=m1_ok,
        reasons=(m15_reason, m5_reason, m1_reason),
    )


def assess_snapshot(
    regime_direction: str,
    snapshot_features: dict[str, Optional[MTFFeatures]],
) -> SetupAssessment:
    """
    Convenience wrapper for a snapshot's already-calculated features.
    """
    return assess_setup(regime_direction, snapshot_features)
