from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .mtf_regime import RegimeAssessment
from .mtf_setup import SetupAssessment


@dataclass(frozen=True)
class MarketConditions:
    """
    Market-level conditions used by the final decision layer.

    These are scoring inputs, not trade execution instructions.
    Risk vetoes belong to a separate risk layer.
    """

    volatility_score: int = 0
    spread_score: int = 0
    session_score: int = 0

    @property
    def total(self) -> int:
        return (
            self.volatility_score
            + self.spread_score
            + self.session_score
        )


@dataclass(frozen=True)
class DecisionAssessment:
    direction: str
    score: int

    regime_score: int
    setup_score: int
    volatility_score: int
    spread_score: int
    session_score: int

    executable: bool
    tier: str
    reasons: tuple[str, ...]

    @property
    def premium(self) -> bool:
        return self.executable and self.score >= 85

    @property
    def strong(self) -> bool:
        return self.executable and self.score >= 75

    @property
    def normal(self) -> bool:
        return self.executable and 65 <= self.score < 75


def _normalize_regime_score(regime: RegimeAssessment) -> int:
    """
    Convert the existing regime score from approximately [-95, +95]
    into a directional strength score from 0 to 35.

    Direction is supplied separately.
    """
    strength = min(abs(regime.score), 95)
    return round((strength / 95.0) * 35)


def _normalize_setup_score(setup: SetupAssessment) -> int:
    """
    Convert setup score from [0, 30] into [0, 50].

    M15 remains mandatory through setup.valid.
    """
    return round((setup.score / 30.0) * 50)


def _clamp_score(value: int) -> int:
    return max(0, min(100, int(value)))


def _tier(score: int, executable: bool) -> str:
    if not executable:
        return "NO_TRADE"

    if score >= 85:
        return "PREMIUM"

    if score >= 75:
        return "STRONG"

    if score >= 65:
        return "NORMAL"

    return "WEAK"


def assess_decision(
    regime: RegimeAssessment,
    setup: SetupAssessment,
    conditions: Optional[MarketConditions] = None,
) -> DecisionAssessment:
    """
    Produce the final 0-100 MTF decision score.

    Score allocation:
        Regime strength       35
        Setup                 50
        Volatility             5
        Spread                 5
        Session                5
        -------------------------
        Total                100

    Hard rule:
        The regime and setup direction must agree.
        M15 setup must be valid.
        A neutral regime cannot produce an executable decision.

    This function does not place orders.
    """
    if conditions is None:
        conditions = MarketConditions()

    reasons: list[str] = []

    direction = regime.direction

    regime_points = _normalize_regime_score(regime)
    setup_points = _normalize_setup_score(setup)

    volatility_points = max(0, min(5, conditions.volatility_score))
    spread_points = max(0, min(5, conditions.spread_score))
    session_points = max(0, min(5, conditions.session_score))

    score = _clamp_score(
        regime_points
        + setup_points
        + volatility_points
        + spread_points
        + session_points
    )

    executable = True

    if direction not in {"LONG", "SHORT"}:
        executable = False
        reasons.append("Higher-timeframe regime is neutral")

    if setup.direction != direction:
        executable = False
        reasons.append("Setup direction conflicts with regime")

    if not setup.m15_valid:
        executable = False
        reasons.append("M15 structure is not valid")

    if not setup.valid:
        executable = False
        reasons.append("Setup is not executable")

    market_quality = (
        volatility_points
        + spread_points
        + session_points
    )

    # Technical alignment alone is not enough.
    # Require a minimum market-quality score before allowing execution.
    if market_quality < 5:
        executable = False
        reasons.append(
            f"Market quality too weak: {market_quality}/15"
        )

    if executable:
        reasons.append(f"{direction} regime and setup aligned")

    reasons.append(f"Regime points: {regime_points}/35")
    reasons.append(f"Setup points: {setup_points}/50")
    reasons.append(f"Volatility points: {volatility_points}/5")
    reasons.append(f"Spread points: {spread_points}/5")
    reasons.append(f"Session points: {session_points}/5")

    return DecisionAssessment(
        direction=direction if executable else "NO_TRADE",
        score=score,
        regime_score=regime_points,
        setup_score=setup_points,
        volatility_score=volatility_points,
        spread_score=spread_points,
        session_score=session_points,
        executable=executable,
        tier=_tier(score, executable),
        reasons=tuple(reasons),
    )
