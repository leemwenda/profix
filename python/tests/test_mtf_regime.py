from profx.mtf_features import MTFFeatures
from profx.mtf_regime import assess_regime


def feature(
    timeframe,
    fast,
    slow,
):
    return MTFFeatures(
        timeframe=timeframe,
        close=100.0,
        atr=2.0,
        ema_fast=fast,
        ema_slow=slow,
        momentum=0.01,
        candle_range=2.0,
        body=1.0,
        body_ratio=0.5,
        close_location=0.75,
    )


def test_all_timeframes_bullish():
    features = {
        "H1": feature("H1", 110, 100),
        "M15": feature("M15", 110, 100),
        "M5": feature("M5", 110, 100),
        "M1": feature("M1", 110, 100),
    }

    regime = assess_regime(features)

    assert regime.direction == "LONG"
    assert regime.bullish
    assert regime.score == 95


def test_all_timeframes_bearish():
    features = {
        "H1": feature("H1", 90, 100),
        "M15": feature("M15", 90, 100),
        "M5": feature("M5", 90, 100),
        "M1": feature("M1", 90, 100),
    }

    regime = assess_regime(features)

    assert regime.direction == "SHORT"
    assert regime.bearish
    assert regime.score == -95


def test_m1_cannot_reverse_h1():
    features = {
        "H1": feature("H1", 110, 100),
        "M15": feature("M15", 110, 100),
        "M5": feature("M5", 110, 100),
        "M1": feature("M1", 90, 100),
    }

    regime = assess_regime(features)

    assert regime.direction == "LONG"
    assert regime.score == 85


def test_h1_conflict_blocks_direction():
    features = {
        "H1": feature("H1", 110, 100),
        "M15": feature("M15", 90, 100),
        "M5": feature("M5", 90, 100),
        "M1": feature("M1", 90, 100),
    }

    regime = assess_regime(features)

    assert regime.direction == "NEUTRAL"
    assert regime.score == 20


def test_missing_lower_timeframes():
    features = {
        "H1": feature("H1", 110, 100),
        "M15": None,
        "M5": None,
        "M1": None,
    }

    regime = assess_regime(features)

    assert regime.direction == "LONG"
    assert regime.score == 50


def test_missing_h1_blocks_trade():
    features = {
        "H1": None,
        "M15": feature("M15", 110, 100),
        "M5": feature("M5", 110, 100),
        "M1": feature("M1", 110, 100),
    }

    regime = assess_regime(features)

    assert regime.direction == "NEUTRAL"
    assert regime.score == 0
    assert "H1 unavailable" in regime.reasons
