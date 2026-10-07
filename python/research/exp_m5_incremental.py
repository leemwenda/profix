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
# Helpers
# ---------------------------------------------------------------------

def ema_trend(features):

    if features is None:
        return None

    if (
        features.ema_fast is None
        or features.ema_slow is None
    ):
        return None

    if features.ema_fast > features.ema_slow:
        return "LONG"

    if features.ema_fast < features.ema_slow:
        return "SHORT"

    return None


def current_m5_confirmation(
    direction,
    features,
):
    if features is None:
        return False

    if (
        features.momentum is None
        or features.close is None
        or features.ema_fast is None
    ):
        return False

    if direction == "LONG":
        return (
            features.momentum > 0
            and features.close > features.ema_fast
        )

    if direction == "SHORT":
        return (
            features.momentum < 0
            and features.close < features.ema_fast
        )

    return False


# ---------------------------------------------------------------------
# Build ONE exact baseline signal set
#
# We then label each executable signal according to two independent
# M5 properties:
#
# A) M5 REGIME ALIGNMENT
#    EMA20/EMA50 agrees with direction.
#
# B) M5 SETUP CONFIRMATION
#    Current production M5 confirmation passes:
#    momentum + close/EMA20.
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

    if direction not in {
        "LONG",
        "SHORT",
    }:
        continue

    if direction == "LONG":
        long_arr[i] = True
    else:
        short_arr[i] = True

    m5 = features["M5"]

    # -------------------------------------------------------------
    # M5 regime alignment
    # -------------------------------------------------------------

    m5_regime_aligned = (
        ema_trend(m5) == direction
    )

    # -------------------------------------------------------------
    # M5 setup confirmation
    # -------------------------------------------------------------

    m5_setup_confirmed = (
        current_m5_confirmation(
            direction,
            m5,
        )
    )

    # -------------------------------------------------------------
    # Regime score already contains M5 information.
    #
    # Current regime weighting:
    # M5 aligned with direction -> +15
    # M5 opposite direction     -> -10
    # neutral                    -> 0
    # -------------------------------------------------------------

    if m5_regime_aligned:
        m5_regime_points = 15

    elif (
        ema_trend(m5) is not None
        and ema_trend(m5) != direction
    ):
        m5_regime_points = -10

    else:
        m5_regime_points = 0

    records.append(
        {
            "decision_time": decision_time,
            "direction": direction,
            "score": decision.score,
            "tier": decision.tier,
            "m5_regime_aligned": m5_regime_aligned,
            "m5_setup_confirmed": m5_setup_confirmed,
            "m5_regime_points": m5_regime_points,
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
# Attach M5 diagnostics to actual trades
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


def fmt_pf(value):

    if np.isnan(value):
        return "N/A"

    if np.isinf(value):
        return "INF"

    return f"{value:.3f}"


# ---------------------------------------------------------------------
# 2x2 M5 matrix
# ---------------------------------------------------------------------

print()
print("=" * 115)
print("M5 INCREMENTAL VALUE ANALYSIS")
print("=" * 115)

print()
print(
    f"Baseline trades : {len(trades)}"
)

print(
    f"Matched M5 rows : "
    f"{trades['m5_setup_confirmed'].notna().sum()}"
)

print()
print(
    f"{'GROUP':30s} "
    f"{'TRADES':>7s} "
    f"{'NET':>10s} "
    f"{'WIN':>8s} "
    f"{'PF':>7s} "
    f"{'AVG':>9s}"
)

print("-" * 115)


groups = [
    (
        "REGIME YES / SETUP YES",
        (
            (trades["m5_regime_aligned"])
            & (trades["m5_setup_confirmed"])
        ),
    ),
    (
        "REGIME YES / SETUP NO",
        (
            (trades["m5_regime_aligned"])
            & (~trades["m5_setup_confirmed"])
        ),
    ),
    (
        "REGIME NO / SETUP YES",
        (
            (~trades["m5_regime_aligned"])
            & (trades["m5_setup_confirmed"])
        ),
    ),
    (
        "REGIME NO / SETUP NO",
        (
            (~trades["m5_regime_aligned"])
            & (~trades["m5_setup_confirmed"])
        ),
    ),
]


for name, mask in groups:

    r = stats(
        trades.loc[mask]
    )

    print(
        f"{name:30s} "
        f"{r['n']:7d} "
        f"${r['net']:9.2f} "
        f"{r['win']:7.2f}% "
        f"{fmt_pf(r['pf']):>7s} "
        f"${r['avg']:8.2f}"
    )


# ---------------------------------------------------------------------
# M5 regime effect
# ---------------------------------------------------------------------

print()
print("=" * 115)
print("M5 REGIME ALIGNMENT EFFECT")
print("=" * 115)

for name, mask in [
    (
        "M5 REGIME ALIGNED",
        trades["m5_regime_aligned"],
    ),
    (
        "M5 REGIME NOT ALIGNED",
        ~trades["m5_regime_aligned"],
    ),
]:

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
# M5 setup effect
# ---------------------------------------------------------------------

print()
print("=" * 115)
print("M5 SETUP CONFIRMATION EFFECT")
print("=" * 115)

for name, mask in [
    (
        "M5 SETUP CONFIRMED",
        trades["m5_setup_confirmed"],
    ),
    (
        "M5 SETUP NOT CONFIRMED",
        ~trades["m5_setup_confirmed"],
    ),
]:

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
# By year
# ---------------------------------------------------------------------

trades["year"] = (
    trades["entry_time"]
    .dt.year
)

print()
print("=" * 115)
print("M5 SETUP CONFIRMATION BY YEAR")
print("=" * 115)

for year, group in trades.groupby(
    "year"
):

    print()
    print(f"YEAR {year}")

    for name, mask in [
        (
            "M5 YES",
            group["m5_setup_confirmed"],
        ),
        (
            "M5 NO",
            ~group["m5_setup_confirmed"],
        ),
    ]:

        r = stats(
            group.loc[mask]
        )

        if r["n"] == 0:
            continue

        print(
            f"  {name:10s} "
            f"trades={r['n']:3d}  "
            f"net=${r['net']:8.2f}  "
            f"win={r['win']:6.2f}%  "
            f"PF={fmt_pf(r['pf'])}"
        )


# ---------------------------------------------------------------------
# Premium relationship
# ---------------------------------------------------------------------

print()
print("=" * 115)
print("PREMIUM TRADES AND M5")
print("=" * 115)

premium = trades[
    trades["score"] >= 85
]

non_premium = trades[
    trades["score"] < 85
]

for name, group in [
    ("PREMIUM 85+", premium),
    ("NON-PREMIUM", non_premium),
]:

    print()
    print(name)

    if group.empty:
        continue

    print(
        f"  trades={len(group)}"
    )

    print(
        f"  M5 regime aligned="
        f"{group['m5_regime_aligned'].mean() * 100:.2f}%"
    )

    print(
        f"  M5 setup confirmed="
        f"{group['m5_setup_confirmed'].mean() * 100:.2f}%"
    )


print()
print("=" * 115)
print("BASELINE CONTROL")
print("=" * 115)

equity = pd.Series(
    result.equity
).astype(float)

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

print(
    f"Trades   : {len(trades)}"
)

print(
    f"Net P/L  : ${float(all_p.sum()):.2f}"
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
    f"PF       : {pf:.3f}"
)

print()
print("=" * 115)
print("IMPORTANT")
print("=" * 115)

print("""
This is a diagnostic experiment.

M5 REGIME ALIGNMENT:
    M5 EMA20/EMA50 agrees with the trade direction.

M5 SETUP CONFIRMATION:
    Current production M5 confirmation:
    momentum agrees with direction AND
    close is on the correct side of EMA20.

The purpose is to determine whether M5's apparent edge
comes from the regime layer, the setup layer, or both.

No production files were modified.
No Git commits were created.
""")
