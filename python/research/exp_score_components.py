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
# Build baseline signals + every score component
# ---------------------------------------------------------------------

long_arr = np.zeros(
    len(h1),
    dtype=bool,
)

short_arr = np.zeros(
    len(h1),
    dtype=bool,
)

component_rows = []


for i, decision_time in enumerate(decision_times):

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

    executable = (
        decision.executable
        and market_ok
    )

    if not executable:
        continue

    if decision.direction not in {
        "LONG",
        "SHORT",
    }:
        continue

    if decision.direction == "LONG":
        long_arr[i] = True
    else:
        short_arr[i] = True

    # -------------------------------------------------------------
    # Reconstruct individual score contributions
    #
    # Total:
    #   Regime       35
    #   M15          25 max
    #   M5           10
    #   M1            5
    #   Volatility    5
    #   Spread        5
    #   Session       5
    # -------------------------------------------------------------

    m15_points = (
        15 if setup.m15_valid else 0
    )

    m5_points = (
        10 if setup.m5_valid else 0
    )

    m1_points = (
        5 if setup.m1_valid else 0
    )

    market_quality = (
        decision.volatility_score
        + decision.spread_score
        + decision.session_score
    )

    component_rows.append(
        {
            "decision_time": decision_time,
            "direction": decision.direction,
            "score": decision.score,
            "tier": decision.tier,
            "regime_points": decision.regime_score,
            "m15_points": m15_points,
            "m5_points": m5_points,
            "m1_points": m1_points,
            "volatility_points": decision.volatility_score,
            "spread_points": decision.spread_score,
            "session_points": decision.session_score,
            "market_quality": market_quality,
        }
    )


components = pd.DataFrame(
    component_rows
)

components = components.set_index(
    "decision_time"
)


# ---------------------------------------------------------------------
# One baseline backtest only
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
# Attach decision components to actual trades
# ---------------------------------------------------------------------

trades["entry_time"] = pd.to_datetime(
    trades["entry_time"],
    utc=True,
)

trades["pnl"] = pd.to_numeric(
    trades["pnl"],
    errors="coerce",
)

component_cols = [
    "direction",
    "score",
    "tier",
    "regime_points",
    "m15_points",
    "m5_points",
    "m1_points",
    "volatility_points",
    "spread_points",
    "session_points",
    "market_quality",
]

component_lookup = components[
    component_cols
]

trades = trades.join(
    component_lookup,
    on="entry_time",
    rsuffix="_component",
)


# ---------------------------------------------------------------------
# Matching check
# ---------------------------------------------------------------------

matched = trades[
    "score"
].notna().sum()

unmatched = (
    trades["score"].isna().sum()
)

print()
print("=" * 120)
print("DECISION SCORE COMPONENT ANALYSIS")
print("=" * 120)

print()
print(
    f"Baseline trades : {len(trades)}"
)

print(
    f"Matched scores  : {matched}"
)

print(
    f"Unmatched       : {unmatched}"
)


# ---------------------------------------------------------------------
# Stats helper
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


def fmt_pf(pf):

    if np.isnan(pf):
        return "N/A"

    if np.isinf(pf):
        return "INF"

    return f"{pf:.3f}"


# ---------------------------------------------------------------------
# Individual components
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("OUTCOME BY COMPONENT PRESENCE")
print("=" * 120)

component_tests = [
    (
        "M5_PRESENT",
        trades["m5_points"] > 0,
    ),
    (
        "M5_ABSENT",
        trades["m5_points"] == 0,
    ),
    (
        "M1_PRESENT",
        trades["m1_points"] > 0,
    ),
    (
        "M1_ABSENT",
        trades["m1_points"] == 0,
    ),
    (
        "HIGH_REGIME",
        trades["regime_points"] >= 25,
    ),
    (
        "LOWER_REGIME",
        trades["regime_points"] < 25,
    ),
    (
        "STRONG_M15",
        trades["m15_points"] == 15,
    ),
    (
        "HIGH_MARKET_QUALITY",
        trades["market_quality"] >= 10,
    ),
    (
        "LOW_MARKET_QUALITY",
        trades["market_quality"] < 10,
    ),
]

for name, mask in component_tests:

    r = stats(
        trades.loc[mask]
    )

    print(
        f"{name:25s} "
        f"trades={r['n']:4d}  "
        f"net=${r['net']:9.2f}  "
        f"win={r['win']:6.2f}%  "
        f"PF={fmt_pf(r['pf']):>7s}  "
        f"avg=${r['avg']:7.2f}"
    )


# ---------------------------------------------------------------------
# M5 + M1 combinations
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("LOWER-TIMEFRAME COMBINATIONS")
print("=" * 120)

combinations = [
    (
        "M5_NO_M1",
        (trades["m5_points"] > 0)
        & (trades["m1_points"] == 0),
    ),
    (
        "NO_M5_M1",
        (trades["m5_points"] == 0)
        & (trades["m1_points"] > 0),
    ),
    (
        "M5_AND_M1",
        (trades["m5_points"] > 0)
        & (trades["m1_points"] > 0),
    ),
    (
        "NO_M5_NO_M1",
        (trades["m5_points"] == 0)
        & (trades["m1_points"] == 0),
    ),
]

for name, mask in combinations:

    r = stats(
        trades.loc[mask]
    )

    print(
        f"{name:20s} "
        f"trades={r['n']:4d}  "
        f"net=${r['net']:9.2f}  "
        f"win={r['win']:6.2f}%  "
        f"PF={fmt_pf(r['pf']):>7s}  "
        f"avg=${r['avg']:7.2f}"
    )


# ---------------------------------------------------------------------
# Score component combinations
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("COMPLETE SCORE STRUCTURES")
print("=" * 120)

structures = [
    (
        "M15_ONLY",
        (trades["m15_points"] > 0)
        & (trades["m5_points"] == 0)
        & (trades["m1_points"] == 0),
    ),
    (
        "M15_M5",
        (trades["m15_points"] > 0)
        & (trades["m5_points"] > 0)
        & (trades["m1_points"] == 0),
    ),
    (
        "M15_M1",
        (trades["m15_points"] > 0)
        & (trades["m5_points"] == 0)
        & (trades["m1_points"] > 0),
    ),
    (
        "M15_M5_M1",
        (trades["m15_points"] > 0)
        & (trades["m5_points"] > 0)
        & (trades["m1_points"] > 0),
    ),
]

for name, mask in structures:

    r = stats(
        trades.loc[mask]
    )

    print(
        f"{name:20s} "
        f"trades={r['n']:4d}  "
        f"net=${r['net']:9.2f}  "
        f"win={r['win']:6.2f}%  "
        f"PF={fmt_pf(r['pf']):>7s}  "
        f"avg=${r['avg']:7.2f}"
    )


# ---------------------------------------------------------------------
# Score buckets with components
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("COMPONENTS INSIDE PREMIUM VS NON-PREMIUM")
print("=" * 120)

for name, mask in [
    (
        "NON_PREMIUM_<85",
        trades["score"] < 85,
    ),
    (
        "PREMIUM_85_PLUS",
        trades["score"] >= 85,
    ),
]:

    group = trades.loc[
        mask
    ]

    r = stats(group)

    print()
    print(name)

    print(
        f"  trades={r['n']}  "
        f"net=${r['net']:.2f}  "
        f"win={r['win']:.2f}%  "
        f"PF={fmt_pf(r['pf'])}"
    )

    if not group.empty:

        for col in (
            "regime_points",
            "m15_points",
            "m5_points",
            "m1_points",
            "market_quality",
        ):

            print(
                f"  avg_{col}="
                f"{group[col].mean():.2f}"
            )


# ---------------------------------------------------------------------
# By year
# ---------------------------------------------------------------------

trades["year"] = (
    trades["entry_time"]
    .dt.year
)

print()
print("=" * 120)
print("M5 PRESENCE BY YEAR")
print("=" * 120)

for year, group in trades.groupby("year"):

    print()
    print(f"YEAR {year}")

    for name, mask in [
        (
            "M5_PRESENT",
            group["m5_points"] > 0,
        ),
        (
            "M5_ABSENT",
            group["m5_points"] == 0,
        ),
    ]:

        r = stats(
            group.loc[mask]
        )

        if r["n"] == 0:
            continue

        print(
            f"  {name:12s} "
            f"trades={r['n']:3d}  "
            f"net=${r['net']:8.2f}  "
            f"win={r['win']:6.2f}%  "
            f"PF={fmt_pf(r['pf'])}"
        )


# ---------------------------------------------------------------------
# Baseline control
# ---------------------------------------------------------------------

equity = pd.Series(
    result.equity
).astype(float)

running_max = equity.cummax()

drawdown = (
    running_max - equity
)

all_p = trades["pnl"].dropna()

wins = all_p[all_p > 0]
losses = all_p[all_p < 0]

gross_win = float(
    wins.sum()
)

gross_loss = abs(
    float(losses.sum())
)

pf = (
    gross_win / gross_loss
    if gross_loss
    else float("inf")
)

print()
print("=" * 120)
print("BASELINE CONTROL")
print("=" * 120)

print(
    f"Trades   : {len(trades)}"
)

print(
    f"Net P/L  : "
    f"${float(all_p.sum()):.2f}"
)

print(
    f"Return   : "
    f"{(float(equity.iloc[-1]) / result.initial_balance - 1) * 100:.2f}%"
)

print(
    f"Win rate : "
    f"{(all_p > 0).mean() * 100:.2f}%"
)

print(
    f"PF       : "
    f"{pf:.3f}"
)

print(
    f"Max DD   : "
    f"${float(drawdown.max()):.2f}"
)


print()
print("=" * 120)
print("IMPORTANT")
print("=" * 120)

print("""
This is a diagnostic experiment.

It runs ONE baseline backtest and attaches the individual
decision-score components to each actual trade.

It does NOT claim that a component caused the outcome.

The purpose is to identify which existing components are
associated with stronger or weaker trade outcomes before
changing the production scoring model.

No production files were modified.
No Git commits were created.
""")
