from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profx.config import load_config
from profx.mtf import load_mtf
from profx.mtf_backtest import (
    _indicator_frame,
    _align_closed_rows,
    _features_from_row,
    _conditions,
)
from profx.mtf_regime import assess_regime
from profx.mtf_setup import assess_setup
from profx.mtf_decision_engine import assess_decision
from profx.engine import MarketData, BacktestEngine


ROOT = Path(__file__).resolve().parents[2]

CFG0 = load_config(ROOT / "config/xauusd.toml")
CFG = CFG0.with_risk(max_drawdown_pct=99.0)

SYMBOL = "XAUUSD"
DATA_DIR = ROOT / "python/data/mtf"

mtf = load_mtf(DATA_DIR, SYMBOL)

h1 = mtf.frame("H1")

decision_times = h1.index + pd.Timedelta(hours=1)


# ---------------------------------------------------------------------
# Prepare aligned CLOSED candles
# ---------------------------------------------------------------------

aligned = {}

for timeframe in ("H1", "M15", "M5", "M1"):

    frame = mtf.frame(timeframe)

    indicators = _indicator_frame(
        frame,
        atr_period=14,
        ema_fast_period=20,
        ema_slow_period=50,
        momentum_period=5,
    )

    _, rows = _align_closed_rows(
        frame,
        indicators,
        decision_times,
        timeframe,
    )

    aligned[timeframe] = rows


# ---------------------------------------------------------------------
# Signal arrays
# ---------------------------------------------------------------------

long_arr = np.zeros(
    len(h1),
    dtype=bool,
)

short_arr = np.zeros(
    len(h1),
    dtype=bool,
)


# ---------------------------------------------------------------------
# Store score models
# ---------------------------------------------------------------------

records = []


for i, decision_time in enumerate(
    decision_times
):

    decision_time = pd.Timestamp(
        decision_time
    ).tz_convert("UTC")

    features = {}

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

    if features["H1"] is None:
        continue

    h1_row = aligned["H1"].iloc[i]

    spread = h1_row.get("spread")

    if pd.isna(spread):
        spread = None
    else:
        spread = float(spread)

    conditions, market_ok = _conditions(
        features["H1"],
        spread,
        decision_time,
        CFG,
        symbol=SYMBOL,
    )

    # -------------------------------------------------------------
    # CURRENT MODEL
    # -------------------------------------------------------------

    current_regime = assess_regime(
        features
    )

    current_setup = assess_setup(
        current_regime.direction,
        features,
    )

    current_decision = assess_decision(
        current_regime,
        current_setup,
        conditions,
    )

    current_executable = (
        current_decision.executable
        and market_ok
    )

    # -------------------------------------------------------------
    # NO M5 REGIME
    #
    # Keep everything identical except M5 is unavailable
    # to the regime layer.
    # -------------------------------------------------------------

    features_no_m5_regime = dict(
        features
    )

    features_no_m5_regime["M5"] = None

    no_m5_regime = assess_regime(
        features_no_m5_regime
    )

    no_m5_regime_decision = assess_decision(
        no_m5_regime,
        current_setup,
        conditions,
    )

    no_m5_regime_executable = (
        no_m5_regime_decision.executable
        and market_ok
    )

    # -------------------------------------------------------------
    # NO M5 SETUP
    #
    # Keep current regime unchanged.
    # Remove M5 only from setup.
    # -------------------------------------------------------------

    features_no_m5_setup = dict(
        features
    )

    features_no_m5_setup["M5"] = None

    no_m5_setup = assess_setup(
        current_regime.direction,
        features_no_m5_setup,
    )

    no_m5_setup_decision = assess_decision(
        current_regime,
        no_m5_setup,
        conditions,
    )

    no_m5_setup_executable = (
        no_m5_setup_decision.executable
        and market_ok
    )

    # -------------------------------------------------------------
    # NO M5 ANYWHERE
    # -------------------------------------------------------------

    no_m5_any_regime = assess_regime(
        features_no_m5_setup
    )

    no_m5_any_setup = assess_setup(
        no_m5_any_regime.direction,
        features_no_m5_setup,
    )

    no_m5_any_decision = assess_decision(
        no_m5_any_regime,
        no_m5_any_setup,
        conditions,
    )

    no_m5_any_executable = (
        no_m5_any_decision.executable
        and market_ok
    )

    # -------------------------------------------------------------
    # Only the CURRENT model determines the actual baseline
    # trade set used below.
    # -------------------------------------------------------------

    if not current_executable:
        continue

    direction = current_decision.direction

    if direction == "LONG":
        long_arr[i] = True
    elif direction == "SHORT":
        short_arr[i] = True
    else:
        continue

    records.append(
        {
            "decision_time": decision_time,

            "current_score": current_decision.score,
            "no_m5_regime_score": no_m5_regime_decision.score,
            "no_m5_setup_score": no_m5_setup_decision.score,
            "no_m5_any_score": no_m5_any_decision.score,

            "current_direction": current_decision.direction,
            "no_m5_regime_direction": no_m5_regime_decision.direction,
            "no_m5_setup_direction": no_m5_setup_decision.direction,
            "no_m5_any_direction": no_m5_any_decision.direction,

            "current_executable": current_executable,
            "no_m5_regime_executable": no_m5_regime_executable,
            "no_m5_setup_executable": no_m5_setup_executable,
            "no_m5_any_executable": no_m5_any_executable,

            "current_regime_points": current_decision.regime_score,
            "no_m5_regime_points": no_m5_regime_decision.regime_score,

            "current_setup_points": current_decision.setup_score,
            "no_m5_setup_points": no_m5_setup_decision.setup_score,

            "m5_setup_valid": current_setup.m5_valid,
            "m5_regime_direction": (
                "LONG"
                if (
                    features["M5"] is not None
                    and features["M5"].ema_fast is not None
                    and features["M5"].ema_slow is not None
                    and features["M5"].ema_fast
                    > features["M5"].ema_slow
                )
                else (
                    "SHORT"
                    if (
                        features["M5"] is not None
                        and features["M5"].ema_fast is not None
                        and features["M5"].ema_slow is not None
                        and features["M5"].ema_fast
                        < features["M5"].ema_slow
                    )
                    else "NEUTRAL"
                )
            ),
        }
    )


lookup = pd.DataFrame(
    records
).set_index(
    "decision_time"
)


# ---------------------------------------------------------------------
# ONE baseline backtest
# ---------------------------------------------------------------------

market = MarketData(
    {"XAUUSD": h1},
    htf_hours=CFG.htf_hours,
    htf_offset=CFG.htf_offset_hours,
)

result = BacktestEngine(
    market,
    CFG,
    signal_override={
        "XAUUSD": (
            long_arr,
            short_arr,
        )
    },
    window=(None, None),
).run()

trades = pd.DataFrame(
    result.trades
)

if trades.empty:
    print("No trades produced.")
    raise SystemExit(0)


# ---------------------------------------------------------------------
# Attach structural scores to actual trades
# ---------------------------------------------------------------------

trades["entry_time"] = pd.to_datetime(
    trades["entry_time"],
    utc=True,
)

trades["pnl"] = pd.to_numeric(
    trades["pnl"],
    errors="coerce",
)

trades = trades.join(
    lookup,
    on="entry_time",
)


# ---------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------

def stats(frame):

    p = frame["pnl"].dropna()

    if p.empty:
        return {
            "n": 0,
            "net": 0.0,
            "win": 0.0,
            "pf": float("nan"),
            "avg": 0.0,
        }

    wins = p[p > 0]
    losses = p[p < 0]

    gross_win = float(
        wins.sum()
    )

    gross_loss = abs(
        float(losses.sum())
    )

    return {
        "n": len(p),
        "net": float(p.sum()),
        "win": float(
            (p > 0).mean() * 100
        ),
        "pf": (
            gross_win / gross_loss
            if gross_loss
            else float("inf")
        ),
        "avg": float(p.mean()),
    }


def fmt_pf(value):

    if np.isnan(value):
        return "N/A"

    if np.isinf(value):
        return "INF"

    return f"{value:.3f}"


# ---------------------------------------------------------------------
# Signal-level effect
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("M5 DOUBLE-COUNTING ABLATION")
print("=" * 125)

print()
print(
    f"Current executable signals : "
    f"{len(lookup)}"
)

for name, column in (
    (
        "NO_M5_REGIME",
        "no_m5_regime_executable",
    ),
    (
        "NO_M5_SETUP",
        "no_m5_setup_executable",
    ),
    (
        "NO_M5_ANYWHERE",
        "no_m5_any_executable",
    ),
):

    count = int(
        lookup[column].sum()
    )

    print(
        f"{name:18s} "
        f"still executable={count:5d} "
        f"changed={len(lookup) - count:5d}"
    )


# ---------------------------------------------------------------------
# Direction changes
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("DIRECTION CHANGES")
print("=" * 125)

for name, column in (
    (
        "NO_M5_REGIME",
        "no_m5_regime_direction",
    ),
    (
        "NO_M5_SETUP",
        "no_m5_setup_direction",
    ),
    (
        "NO_M5_ANYWHERE",
        "no_m5_any_direction",
    ),
):

    changed = (
        lookup[column]
        != lookup["current_direction"]
    )

    print(
        f"{name:18s} "
        f"direction_changed="
        f"{int(changed.sum())}"
    )


# ---------------------------------------------------------------------
# Score movement
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("SCORE MOVEMENT")
print("=" * 125)

models = [
    (
        "NO_M5_REGIME",
        "no_m5_regime_score",
    ),
    (
        "NO_M5_SETUP",
        "no_m5_setup_score",
    ),
    (
        "NO_M5_ANYWHERE",
        "no_m5_any_score",
    ),
]

for name, column in models:

    delta = (
        lookup[column]
        - lookup["current_score"]
    )

    print(
        f"{name:18s} "
        f"mean_delta={delta.mean():+.2f}  "
        f"median_delta={delta.median():+.2f}  "
        f"min={delta.min():+.0f}  "
        f"max={delta.max():+.0f}"
    )


# ---------------------------------------------------------------------
# Current vs no-M5 score buckets
# ---------------------------------------------------------------------

def score_bucket(score):

    if score < 65:
        return "00-64"

    if score < 75:
        return "65-74"

    if score < 85:
        return "75-84"

    return "85-100"


for score_column, model_name in [
    (
        "current_score",
        "CURRENT",
    ),
    (
        "no_m5_regime_score",
        "NO_M5_REGIME",
    ),
    (
        "no_m5_setup_score",
        "NO_M5_SETUP",
    ),
    (
        "no_m5_any_score",
        "NO_M5_ANYWHERE",
    ),
]:

    trades[
        f"{model_name}_bucket"
    ] = trades[
        score_column
    ].apply(
        score_bucket
    )


# ---------------------------------------------------------------------
# Outcome under structural score models
#
# Important: actual P/L is unchanged because this is ONE baseline
# backtest. We are measuring how the SAME trades are classified.
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("ACTUAL TRADE OUTCOME BY STRUCTURAL SCORE MODEL")
print("=" * 125)

for score_column, model_name in [
    (
        "current_score",
        "CURRENT",
    ),
    (
        "no_m5_regime_score",
        "NO_M5_REGIME",
    ),
    (
        "no_m5_setup_score",
        "NO_M5_SETUP",
    ),
    (
        "no_m5_any_score",
        "NO_M5_ANYWHERE",
    ),
]:

    print()
    print(model_name)

    bucket_column = f"{model_name}_bucket"

    for bucket in (
        "00-64",
        "65-74",
        "75-84",
        "85-100",
    ):

        group = trades[
            trades[bucket_column]
            == bucket
        ]

        r = stats(group)

        print(
            f"  {bucket:10s} "
            f"trades={r['n']:4d}  "
            f"net=${r['net']:9.2f}  "
            f"win={r['win']:6.2f}%  "
            f"PF={fmt_pf(r['pf']):>7s}  "
            f"avg=${r['avg']:7.2f}"
        )


# ---------------------------------------------------------------------
# Premium classification overlap
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("PREMIUM CLASSIFICATION OVERLAP")
print("=" * 125)

current_premium = (
    trades["current_score"]
    >= 85
)

for name, column in (
    (
        "NO_M5_REGIME",
        "no_m5_regime_score",
    ),
    (
        "NO_M5_SETUP",
        "no_m5_setup_score",
    ),
    (
        "NO_M5_ANYWHERE",
        "no_m5_any_score",
    ),
):

    alternative_premium = (
        trades[column]
        >= 85
    )

    both = (
        current_premium
        & alternative_premium
    )

    lost = (
        current_premium
        & ~alternative_premium
    )

    gained = (
        ~current_premium
        & alternative_premium
    )

    print()
    print(name)

    print(
        f"  current premium : "
        f"{int(current_premium.sum())}"
    )

    print(
        f"  both premium    : "
        f"{int(both.sum())}"
    )

    print(
        f"  premium lost    : "
        f"{int(lost.sum())}"
    )

    print(
        f"  premium gained  : "
        f"{int(gained.sum())}"
    )


# ---------------------------------------------------------------------
# Outcome of premium trades under each model
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("PREMIUM TRADE QUALITY")
print("=" * 125)

for name, column in (
    (
        "CURRENT",
        "current_score",
    ),
    (
        "NO_M5_REGIME",
        "no_m5_regime_score",
    ),
    (
        "NO_M5_SETUP",
        "no_m5_setup_score",
    ),
    (
        "NO_M5_ANYWHERE",
        "no_m5_any_score",
    ),
):

    group = trades[
        trades[column]
        >= 85
    ]

    r = stats(group)

    print(
        f"{name:18s} "
        f"trades={r['n']:4d}  "
        f"net=${r['net']:9.2f}  "
        f"win={r['win']:6.2f}%  "
        f"PF={fmt_pf(r['pf']):>7s}  "
        f"avg=${r['avg']:7.2f}"
    )


# ---------------------------------------------------------------------
# M5 layer combinations
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("M5 LAYER COMBINATIONS")
print("=" * 125)

for name, mask in [
    (
        "REGIME + SETUP",
        (
            trades["m5_regime_direction"]
            == trades["current_direction"]
        )
        & trades["m5_setup_valid"],
    ),
    (
        "REGIME ONLY",
        (
            trades["m5_regime_direction"]
            == trades["current_direction"]
        )
        & ~trades["m5_setup_valid"],
    ),
    (
        "SETUP ONLY",
        (
            trades["m5_regime_direction"]
            != trades["current_direction"]
        )
        & trades["m5_setup_valid"],
    ),
    (
        "NEITHER",
        (
            trades["m5_regime_direction"]
            != trades["current_direction"]
        )
        & ~trades["m5_setup_valid"],
    ),
]:

    r = stats(
        trades.loc[mask]
    )

    print(
        f"{name:18s} "
        f"trades={r['n']:4d}  "
        f"net=${r['net']:9.2f}  "
        f"win={r['win']:6.2f}%  "
        f"PF={fmt_pf(r['pf']):>7s}  "
        f"avg=${r['avg']:7.2f}"
    )


# ---------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------

all_p = trades[
    "pnl"
].dropna()

wins = all_p[
    all_p > 0
]

losses = all_p[
    all_p < 0
]

gross_loss = abs(
    float(losses.sum())
)

pf = (
    float(wins.sum())
    / gross_loss
    if gross_loss
    else float("inf")
)

print()
print("=" * 125)
print("BASELINE CONTROL")
print("=" * 125)

print(
    f"Trades   : {len(all_p)}"
)

print(
    f"Net P/L  : ${float(all_p.sum()):.2f}"
)

print(
    f"Win rate : "
    f"{float((all_p > 0).mean() * 100):.2f}%"
)

print(
    f"PF       : {pf:.3f}"
)

print()
print("=" * 125)
print("IMPORTANT")
print("=" * 125)

print("""
This is a structural ablation.

CURRENT:
    M5 contributes to both:
        1. H1 regime scoring
        2. M5 setup scoring

NO_M5_REGIME:
    M5 is removed only from assess_regime().

NO_M5_SETUP:
    M5 is removed only from assess_setup().

NO_M5_ANYWHERE:
    M5 is removed from both layers.

All four models are evaluated against the SAME baseline
trade outcomes. This isolates how M5 changes the scoring
and classification without creating separate path-dependent
backtests.

No production files were modified.
No Git commits were created.
""")
