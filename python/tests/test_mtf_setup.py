from __future__ import annotations

from profx.mtf_setup import assess_setup
from profx.mtf_features import MTFFeatures


def make_features(
    *,
    timeframe="M15",
    close=100.0,
    atr=2.0,
    ema_fast=99.0,
    ema_slow=98.0,
    momentum=1.0,
    candle_range=2.0,
    body=1.0,
    body_ratio=0.5,
    close_location=0.75,
):
    return MTFFeatures(
        timeframe=timeframe,
        close=close,
        atr=atr,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        momentum=momentum,
        candle_range=candle_range,
        body=body,
        body_ratio=body_ratio,
        close_location=close_location,
    )


def test_long_all_timeframes_valid():
    features = {
        "M15": make_features(),
        "M5": make_features(timeframe="M5"),
        "M1": make_features(timeframe="M1"),
    }

    result = assess_setup("LONG", features)

    assert result.direction == "LONG"
    assert result.score == 30
    assert result.m15_valid
    assert result.m5_valid
    assert result.m1_valid
    assert result.valid


def test_short_all_timeframes_valid():
    features = {
        "M15": MTFFeatures(
            timeframe="M15",
            close=95,
            atr=2,
            ema_fast=96,
            ema_slow=97,
            momentum=-1,
            candle_range=2,
            body=-1,
            body_ratio=0.5,
            close_location=0.25,
        ),
        "M5": MTFFeatures(
            timeframe="M5",
            close=95,
            atr=2,
            ema_fast=96,
            ema_slow=97,
            momentum=-1,
            candle_range=2,
            body=-1,
            body_ratio=0.5,
            close_location=0.25,
        ),
        "M1": MTFFeatures(
            timeframe="M1",
            close=95,
            atr=2,
            ema_fast=96,
            ema_slow=97,
            momentum=-1,
            candle_range=2,
            body=-1,
            body_ratio=0.5,
            close_location=0.25,
        ),
    }

    result = assess_setup("SHORT", features)

    assert result.direction == "SHORT"
    assert result.score == 30
    assert result.m15_valid
    assert result.m5_valid
    assert result.m1_valid


def test_neutral_regime_cannot_create_setup():
    features = {
        "M15": make_features(),
        "M5": make_features(),
        "M1": make_features(),
    }

    result = assess_setup("NEUTRAL", features)

    assert result.direction == "NEUTRAL"
    assert result.score == 0
    assert not result.valid


def test_m15_is_required_for_valid_setup():
    features = {
        "M15": None,
        "M5": make_features(),
        "M1": make_features(),
    }

    result = assess_setup("LONG", features)

    assert result.direction == "LONG"
    assert result.score == 15
    assert not result.m15_valid
    assert result.m5_valid
    assert result.m1_valid
    assert not result.valid


def test_m5_can_be_missing_without_changing_direction():
    features = {
        "M15": make_features(),
        "M5": None,
        "M1": make_features(),
    }

    result = assess_setup("LONG", features)

    assert result.direction == "LONG"
    assert result.score == 20
    assert result.m15_valid
    assert not result.m5_valid
    assert result.m1_valid
    assert result.valid


def test_m1_is_confirmation_not_authority():
    features = {
        "M15": make_features(),
        "M5": make_features(),
        "M1": None,
    }

    result = assess_setup("LONG", features)

    assert result.direction == "LONG"
    assert result.score == 25
    assert result.valid


def test_m1_cannot_reverse_long_regime():
    features = {
        "M15": make_features(),
        "M5": make_features(),
        "M1": MTFFeatures(
            timeframe="M1",
            close=95,
            atr=2,
            ema_fast=96,
            ema_slow=97,
            momentum=-1,
            candle_range=2,
            body=-1,
            body_ratio=0.5,
            close_location=0.25,
        ),
    }

    result = assess_setup("LONG", features)

    assert result.direction == "LONG"
    assert result.score == 25
    assert not result.m1_valid


def test_m1_cannot_reverse_short_regime():
    features = {
        "M15": MTFFeatures(
            timeframe="M15",
            close=95,
            atr=2,
            ema_fast=96,
            ema_slow=97,
            momentum=-1,
            candle_range=2,
            body=-1,
            body_ratio=0.5,
            close_location=0.25,
        ),
        "M5": MTFFeatures(
            timeframe="M5",
            close=95,
            atr=2,
            ema_fast=96,
            ema_slow=97,
            momentum=-1,
            candle_range=2,
            body=-1,
            body_ratio=0.5,
            close_location=0.25,
        ),
        "M1": make_features(),
    }

    result = assess_setup("SHORT", features)

    assert result.direction == "SHORT"
    assert result.score == 25
    assert not result.m1_valid


def test_weak_m15_does_not_count():
    features = {
        "M15": MTFFeatures(
            timeframe="M15",
            close=100,
            atr=2,
            ema_fast=99,
            ema_slow=101,
            momentum=1,
            candle_range=2,
            body=1,
            body_ratio=0.5,
            close_location=0.75,
        ),
        "M5": make_features(),
        "M1": make_features(),
    }

    result = assess_setup("LONG", features)

    assert result.direction == "LONG"
    assert result.score == 15
    assert not result.m15_valid
    assert result.m5_valid
    assert result.m1_valid


def test_insufficient_indicator_warmup_does_not_count():
    features = {
        "M15": MTFFeatures(
            timeframe="M15",
            close=100,
            atr=None,
            ema_fast=None,
            ema_slow=None,
            momentum=None,
            candle_range=2,
            body=1,
            body_ratio=0.5,
            close_location=0.75,
        ),
        "M5": make_features(),
        "M1": make_features(),
    }

    result = assess_setup("LONG", features)

    assert result.score == 15
    assert not result.m15_valid
    assert result.m5_valid
    assert result.m1_valid
