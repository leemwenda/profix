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
# Build exact executable baseline signals AND score map
# ---------------------------------------------------------------------

long_arr = np.zeros(len(h1), dtype=bool)
short_arr = np.zeros(len(h1), dtype=bool)

score_arr = np.full(len(h1), np.nan)

tier_arr = np.full(
    len(h1),
    "NO_TRADE",
    dtype=object,
)

direction_arr = np.full(
    len(h1),
    "NO_TRADE",
    dtype=object,
)

executable_count = 0


for i, decision_time in enumerate(decision_times):

    decision_time = pd.Timestamp(
        decision_time
    ).tz_convert("UTC")

    features = {}

    for timeframe in ("H1", "M15", "M5", "M1"):
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

    direction = decision.direction

    if direction not in {"LONG", "SHORT"}:
        continue

    executable_count += 1

    if direction == "LONG":
        long_arr[i] = True
    else:
        short_arr[i] = True

    score_arr[i] = decision.score
    tier_arr[i] = decision.tier
    direction_arr[i] = direction


# ---------------------------------------------------------------------
# Score lookup
# ---------------------------------------------------------------------

score_table = pd.DataFrame(
    {
        "decision_time": pd.DatetimeIndex(
            decision_times
        ).tz_convert("UTC"),
        "score": score_arr,
        "tier": tier_arr,
        "direction": direction_arr,
    }
)

score_table = score_table[
    np.isfinite(score_table["score"])
].copy()

score_table["score"] = (
    score_table["score"]
    .astype(int)
)

score_table = score_table.set_index(
    "decision_time"
)


# ---------------------------------------------------------------------
# Run ONE baseline backtest
#
# We intentionally do not run separate backtests for each bucket.
# That avoids path-dependent distortion between score groups.
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


trades = pd.DataFrame(result.trades)

if trades.empty:
    print("No trades were produced.")
    raise SystemExit(0)


# ---------------------------------------------------------------------
# Attach score to actual entry time
# ---------------------------------------------------------------------

trades["entry_time"] = pd.to_datetime(
    trades["entry_time"],
    utc=True,
)

trades["pnl"] = pd.to_numeric(
    trades["pnl"],
    errors="coerce",
)

trades["score"] = trades["entry_time"].map(
    score_table["score"]
)

trades["score_tier"] = trades["entry_time"].map(
    score_table["tier"]
)

trades["direction"] = trades["entry_time"].map(
    score_table["direction"]
)


# ---------------------------------------------------------------------
# Check score matching
# ---------------------------------------------------------------------

matched = trades["score"].notna().sum()
unmatched = trades["score"].isna().sum()

print()
print("=" * 110)
print("DECISION SCORE CALIBRATION")
print("=" * 110)

print()
print(
    f"Executable baseline signals : "
    f"{executable_count}"
)

print(
    f"Backtest trades              : "
    f"{len(trades)}"
)

print(
    f"Trades matched to score      : "
    f"{matched}"
)

print(
    f"Trades NOT matched           : "
    f"{unmatched}"
)

if unmatched:
    print()
    print("UNMATCHED ENTRY TIMES:")
    print(
        trades.loc[
            trades["score"].isna(),
            "entry_time"
        ].head(20).to_string(index=False)
    )


# ---------------------------------------------------------------------
# Score buckets
# ---------------------------------------------------------------------

def bucket(score):

    if score < 65:
        return "00-64"

    if score < 75:
        return "65-74"

    if score < 85:
        return "75-84"

    return "85-100"


trades["score_bucket"] = trades["score"].apply(
    lambda x: bucket(x)
    if pd.notna(x)
    else "UNMATCHED"
)


# ---------------------------------------------------------------------
# Statistics helper
# ---------------------------------------------------------------------

def stats(frame):

    p = frame["pnl"].dropna()

    if p.empty:
        return {
            "trades": 0,
            "net": 0.0,
            "win": 0.0,
            "pf": float("nan"),
            "avg": 0.0,
        }

    wins = p[p > 0]
    losses = p[p < 0]

    gross_win = float(wins.sum())
    gross_loss = abs(float(losses.sum()))

    return {
        "trades": len(p),
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


# ---------------------------------------------------------------------
# Overall score buckets
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("RESULTS BY SCORE BUCKET")
print("=" * 110)

order = [
    "00-64",
    "65-74",
    "75-84",
    "85-100",
]

for name in order:

    group = trades[
        trades["score_bucket"] == name
    ]

    r = stats(group)

    pf = r["pf"]

    if np.isnan(pf):
        pf_text = "N/A"
    elif np.isinf(pf):
        pf_text = "INF"
    else:
        pf_text = f"{pf:.3f}"

    print(
        f"{name:10s} "
        f"trades={r['trades']:4d}  "
        f"net=${r['net']:9.2f}  "
        f"win={r['win']:6.2f}%  "
        f"PF={pf_text:>7s}  "
        f"avg=${r['avg']:7.2f}"
    )


# ---------------------------------------------------------------------
# Results by exact score
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("RESULTS BY EXACT DECISION SCORE")
print("=" * 110)

exact = []

for score, group in trades[
    trades["score"].notna()
].groupby("score"):

    r = stats(group)

    exact.append(
        (
            int(score),
            r["trades"],
            r["net"],
            r["win"],
            r["pf"],
            r["avg"],
        )
    )

for (
    score,
    n,
    net,
    win,
    pf,
    avg,
) in exact:

    if np.isnan(pf):
        pf_text = "N/A"
    elif np.isinf(pf):
        pf_text = "INF"
    else:
        pf_text = f"{pf:.3f}"

    print(
        f"score={score:3d}  "
        f"trades={n:3d}  "
        f"net=${net:9.2f}  "
        f"win={win:6.2f}%  "
        f"PF={pf_text:>7s}  "
        f"avg=${avg:7.2f}"
    )


# ---------------------------------------------------------------------
# Results by year
# ---------------------------------------------------------------------

trades["year"] = (
    trades["entry_time"]
    .dt.year
)

print()
print("=" * 110)
print("SCORE BUCKETS BY YEAR")
print("=" * 110)

for year, year_group in trades[
    trades["score"].notna()
].groupby("year"):

    print()
    print(f"YEAR {year}")

    for name in order:

        group = year_group[
            year_group["score_bucket"] == name
        ]

        r = stats(group)

        if r["trades"] == 0:
            continue

        pf = r["pf"]

        if np.isnan(pf):
            pf_text = "N/A"
        elif np.isinf(pf):
            pf_text = "INF"
        else:
            pf_text = f"{pf:.3f}"

        print(
            f"  {name:10s} "
            f"trades={r['trades']:3d}  "
            f"net=${r['net']:8.2f}  "
            f"win={r['win']:6.2f}%  "
            f"PF={pf_text}"
        )


# ---------------------------------------------------------------------
# Correlation between score and trade P/L
# ---------------------------------------------------------------------

matched_trades = trades[
    trades["score"].notna()
].copy()

if len(matched_trades) >= 2:

    corr = matched_trades[
        ["score", "pnl"]
    ].corr().iloc[0, 1]

    print()
    print("=" * 110)
    print("SCORE / P&L RELATIONSHIP")
    print("=" * 110)

    print(
        f"Pearson correlation "
        f"(score vs trade P/L): {corr:.4f}"
    )


# ---------------------------------------------------------------------
# Current tier distribution
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("TRADE DISTRIBUTION BY CURRENT DECISION TIER")
print("=" * 110)

tier_counts = (
    trades[
        trades["score_tier"].notna()
    ]["score_tier"]
    .value_counts()
)

for tier in (
    "NORMAL",
    "STRONG",
    "PREMIUM",
    "WEAK",
):

    print(
        f"{tier:10s}: "
        f"{int(tier_counts.get(tier, 0))}"
    )


# ---------------------------------------------------------------------
# Final baseline metrics
# ---------------------------------------------------------------------

equity = pd.Series(
    result.equity
).astype(float)

running_max = equity.cummax()

drawdown = (
    running_max
    - equity
)

pnl = (
    float(equity.iloc[-1])
    - result.initial_balance
)

all_p = trades["pnl"].dropna()

gross_win = float(
    all_p[all_p > 0].sum()
)

gross_loss = abs(
    float(all_p[all_p < 0].sum())
)

pf = (
    gross_win / gross_loss
    if gross_loss
    else float("inf")
)

print()
print("=" * 110)
print("BASELINE CONTROL")
print("=" * 110)

print(
    f"Trades   : {len(trades)}"
)

print(
    f"Net P/L  : ${pnl:.2f}"
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
print("=" * 110)
print("IMPORTANT")
print("=" * 110)

print("""
This experiment runs ONE baseline backtest and tags each actual
trade with the decision score that generated it.

It does not run separate score-bucket backtests, avoiding
path-dependent differences between buckets.

Score buckets:
    00-64
    65-74
    75-84
    85-100

The objective is to determine whether the existing decision score
is properly calibrated.

No production files were modified.
No Git commits were created.
""")
