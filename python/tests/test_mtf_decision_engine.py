from __future__ import annotations

from profx.mtf_decision_engine import (
    MarketConditions,
    assess_decision,
)
from profx.mtf_regime import RegimeAssessment
from profx.mtf_setup import SetupAssessment


def regime(direction="LONG", score=95):
    return RegimeAssessment(
        direction=direction,
        score=score,
        h1_bias=direction,
        m15_bias=direction,
        m5_bias=direction,
        m1_bias=direction,
        reasons=("test",),
    )


def setup(
    direction="LONG",
    score=30,
    m15=True,
    m5=True,
    m1=True,
):
    return SetupAssessment(
        direction=direction,
        score=score,
        m15_valid=m15,
        m5_valid=m5,
        m1_valid=m1,
        reasons=("test",),
    )


def conditions():
    return MarketConditions(
        volatility_score=5,
        spread_score=5,
        session_score=5,
    )


def test_perfect_long_setup_is_premium():
    result = assess_decision(
        regime("LONG", 95),
        setup("LONG", 30, True, True, True),
        conditions(),
    )

    assert result.direction == "LONG"
    assert result.score == 100
    assert result.executable
    assert result.tier == "PREMIUM"
    assert result.premium


def test_perfect_short_setup_is_premium():
    result = assess_decision(
        regime("SHORT", -95),
        setup("SHORT", 30, True, True, True),
        conditions(),
    )

    assert result.direction == "SHORT"
    assert result.score == 100
    assert result.executable
    assert result.tier == "PREMIUM"


def test_neutral_regime_cannot_trade():
    result = assess_decision(
        regime("NEUTRAL", 0),
        setup("LONG", 30, True, True, True),
        conditions(),
    )

    assert result.direction == "NO_TRADE"
    assert not result.executable
    assert result.tier == "NO_TRADE"
    assert result.score == 65


def test_conflicting_setup_cannot_trade():
    result = assess_decision(
        regime("LONG", 95),
        setup("SHORT", 30, True, True, True),
        conditions(),
    )

    assert result.direction == "NO_TRADE"
    assert not result.executable
    assert "Setup direction conflicts with regime" in result.reasons


def test_missing_m15_cannot_trade():
    result = assess_decision(
        regime("LONG", 95),
        setup("LONG", 15, False, True, True),
        conditions(),
    )

    assert result.direction == "NO_TRADE"
    assert not result.executable
    assert "M15 structure is not valid" in result.reasons


def test_m1_missing_does_not_destroy_direction():
    result = assess_decision(
        regime("LONG", 95),
        setup("LONG", 25, True, True, False),
        conditions(),
    )

    assert result.direction == "LONG"
    assert result.executable
    assert result.setup_score == round(25 / 30 * 50)


def test_m5_missing_does_not_destroy_direction():
    result = assess_decision(
        regime("LONG", 95),
        setup("LONG", 20, True, False, False),
        conditions(),
    )

    assert result.direction == "LONG"
    assert result.executable


def test_score_never_exceeds_100():
    result = assess_decision(
        regime("LONG", 95),
        setup("LONG", 30, True, True, True),
        MarketConditions(100, 100, 100),
    )

    assert result.score == 100


def test_negative_market_scores_are_clamped():
    result = assess_decision(
        regime("LONG", 95),
        setup("LONG", 30, True, True, True),
        MarketConditions(-100, -100, -100),
    )

    assert result.score == 85
    assert not result.executable
    assert result.tier == "NO_TRADE"
    assert any("Market quality too weak" in r for r in result.reasons)


def test_75_is_strong():
    result = assess_decision(
        regime("LONG", 95),
        setup("LONG", 24, True, True, True),
        MarketConditions(2, 2, 1),
    )

    assert result.score == 80
    assert result.tier == "STRONG"


def test_65_is_normal():
    result = assess_decision(
        regime("LONG", 95),
        setup("LONG", 18, True, False, False),
        MarketConditions(2, 2, 1),
    )

    assert result.score == 70
    assert result.tier == "NORMAL"


def test_64_is_weak():
    result = assess_decision(
        regime("LONG", 70),
        setup("LONG", 18, True, False, False),
        MarketConditions(2, 2, 1),
    )

    assert result.score == 61
    assert result.tier == "WEAK"
    assert result.executable


def test_direction_follows_regime_not_m1():
    result = assess_decision(
        regime("LONG", 95),
        setup("LONG", 30, True, True, False),
        conditions(),
    )

    assert result.direction == "LONG"


def test_all_score_components_are_reported():
    result = assess_decision(
        regime("LONG", 95),
        setup("LONG", 30, True, True, True),
        MarketConditions(4, 3, 2),
    )

    assert result.regime_score == 35
    assert result.setup_score == 50
    assert result.volatility_score == 4
    assert result.spread_score == 3
    assert result.session_score == 2
    assert result.score == 94


def test_setup_neutral_cannot_trade_against_long_regime():
    result = assess_decision(
        regime("LONG", 95),
        setup("NEUTRAL", 0, False, False, False),
        conditions(),
    )

    assert result.direction == "NO_TRADE"
    assert not result.executable


def test_no_order_execution_is_present():
    """
    Architecture guard: decision layer returns an assessment only.
    """
    result = assess_decision(
        regime("LONG", 95),
        setup("LONG", 30, True, True, True),
        conditions(),
    )

    assert hasattr(result, "score")
    assert not hasattr(result, "place_order")
    assert not hasattr(result, "execute_order")
