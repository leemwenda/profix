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

for timeframe in (
    "H1",
    "M15",
    "M5",
    "M1",
):

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
# Build exact baseline trades and two scores
# ---------------------------------------------------------------------

long_arr = np.zeros(
    len(h1),
    dtype=bool,
)

short_arr = np.zeros(
    len(h1),
    dtype=bool,
)

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

    regime = assess_regime(features)

    setup = assess_setup(
        regime.direction,
        features,
    )

    decision = assess_decision(
        regime,
        setup,
        conditions,
    )

    if not (
        decision.executable
        and market_ok
    ):
        continue

    direction = decision.direction

    if direction == "LONG":
        long_arr[i] = True
    elif direction == "SHORT":
        short_arr[i] = True
    else:
        continue

    # -------------------------------------------------------------
    # CURRENT SCORE
    # -------------------------------------------------------------

    current_score = int(
        decision.score
    )

    # -------------------------------------------------------------
    # Reconstruct setup without M1
    #
    # Current setup:
    # M15 = 15
    # M5  = 10
    # M1  = 5
    #
    # For the no-M1 model, preserve the same 50-point
    # setup allocation but redistribute it across the
    # remaining M15 + M5 components.
    #
    # Remaining maximum = 25.
    # -------------------------------------------------------------

    setup_without_m1 = (
        setup.score
        - (5 if setup.m1_valid else 0)
    )

    setup_without_m1 = max(
        0,
        setup_without_m1,
    )

    no_m1_setup_points = round(
        (
            setup_without_m1
            / 25.0
        ) * 50
    )

    no_m1_score = max(
        0,
        min(
            100,
            int(
                decision.regime_score
                + no_m1_setup_points
                + decision.volatility_score
                + decision.spread_score
                + decision.session_score
            ),
        ),
    )

    records.append(
        {
            "decision_time": decision_time,
            "current_score": current_score,
            "no_m1_score": no_m1_score,
            "m1_valid": setup.m1_valid,
            "direction": direction,
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
# Statistics
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


def bucket(score):

    if score < 65:
        return "00-64"

    if score < 75:
        return "65-74"

    if score < 85:
        return "75-84"

    return "85-100"


def fmt_pf(value):

    if np.isnan(value):
        return "N/A"

    if np.isinf(value):
        return "INF"

    return f"{value:.3f}"


trades["current_bucket"] = (
    trades["current_score"]
    .apply(bucket)
)

trades["no_m1_bucket"] = (
    trades["no_m1_score"]
    .apply(bucket)
)


# ---------------------------------------------------------------------
# Score distribution changes
# ---------------------------------------------------------------------

print()
print("=" * 115)
print("M1 SCORE IMPACT ANALYSIS")
print("=" * 115)

print()
print(
    f"Baseline trades : {len(trades)}"
)

print(
    f"M1-confirmed    : "
    f"{int(trades['m1_valid'].sum())}"
)

print()
print(
    f"{'BUCKET':10s} "
    f"{'CURRENT N':>12s} "
    f"{'NO-M1 N':>12s}"
)

print("-" * 45)

for name in (
    "00-64",
    "65-74",
    "75-84",
    "85-100",
):

    current_n = int(
        (
            trades["current_bucket"]
            == name
        ).sum()
    )

    no_m1_n = int(
        (
            trades["no_m1_bucket"]
            == name
        ).sum()
    )

    print(
        f"{name:10s} "
        f"{current_n:12d} "
        f"{no_m1_n:12d}"
    )


# ---------------------------------------------------------------------
# Current score calibration
# ---------------------------------------------------------------------

print()
print("=" * 115)
print("CURRENT SCORE OUTCOMES")
print("=" * 115)

for name in (
    "00-64",
    "65-74",
    "75-84",
    "85-100",
):

    r = stats(
        trades[
            trades["current_bucket"]
            == name
        ]
    )

    print(
        f"{name:10s} "
        f"trades={r['n']:4d}  "
        f"net=${r['net']:9.2f}  "
        f"win={r['win']:6.2f}%  "
        f"PF={fmt_pf(r['pf']):>7s}  "
        f"avg=${r['avg']:7.2f}"
    )


# ---------------------------------------------------------------------
# No-M1 score calibration
# ---------------------------------------------------------------------

print()
print("=" * 115)
print("NO-M1 SCORE OUTCOMES")
print("=" * 115)

for name in (
    "00-64",
    "65-74",
    "75-84",
    "85-100",
):

    r = stats(
        trades[
            trades["no_m1_bucket"]
            == name
        ]
    )

    print(
        f"{name:10s} "
        f"trades={r['n']:4d}  "
        f"net=${r['net']:9.2f}  "
        f"win={r['win']:6.2f}%  "
        f"PF={fmt_pf(r['pf']):>7s}  "
        f"avg=${r['avg']:7.2f}"
    )


# ---------------------------------------------------------------------
# M1 score movement
# ---------------------------------------------------------------------

print()
print("=" * 115)
print("SCORE MOVEMENT CAUSED BY M1")
print("=" * 115)

trades["score_delta"] = (
    trades["current_score"]
    - trades["no_m1_score"]
)

for value, group in trades.groupby(
    "score_delta"
):

    print(
        f"M1 score contribution "
        f"{value:+3d}  "
        f"trades={len(group):4d}  "
        f"net=${group['pnl'].sum():9.2f}  "
        f"avg=${group['pnl'].mean():7.2f}"
    )


# ---------------------------------------------------------------------
# Important cases:
# trades moved into PREMIUM because of M1
# ---------------------------------------------------------------------

moved_into_premium = (
    (trades["current_score"] >= 85)
    & (trades["no_m1_score"] < 85)
)

print()
print("=" * 115)
print("TRADES MADE PREMIUM BY M1")
print("=" * 115)

group = trades[
    moved_into_premium
]

r = stats(group)

print(
    f"trades={r['n']}  "
    f"net=${r['net']:.2f}  "
    f"win={r['win']:.2f}%  "
    f"PF={fmt_pf(r['pf'])}  "
    f"avg=${r['avg']:.2f}"
)


# ---------------------------------------------------------------------
# Year analysis
# ---------------------------------------------------------------------

trades["year"] = (
    trades["entry_time"]
    .dt.year
)

print()
print("=" * 115)
print("M1 SCORE CONTRIBUTION BY YEAR")
print("=" * 115)

for year, group in trades.groupby(
    "year"
):

    changed = group[
        group["score_delta"] > 0
    ]

    if changed.empty:
        print(
            f"YEAR {year}: "
            f"no trades received M1 points"
        )
        continue

    r = stats(changed)

    print(
        f"YEAR {year}: "
        f"M1-added trades={r['n']:3d}  "
        f"net=${r['net']:8.2f}  "
        f"win={r['win']:6.2f}%  "
        f"PF={fmt_pf(r['pf'])}"
    )


print()
print("=" * 115)
print("IMPORTANT")
print("=" * 115)

print("""
This experiment does NOT remove M1 from execution.

The exact same baseline trades are used throughout.

It only asks:

    Does adding the M1 component to the decision score
    improve or weaken the score's ability to rank trades?

CURRENT:
    M1 contributes to the existing normalized setup score.

NO-M1:
    M1 is removed and the remaining M15 + M5 setup
    is renormalized to the same 50-point setup allocation.

No production files were modified.
No Git commits were created.
""")
