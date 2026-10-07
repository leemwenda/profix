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
# ONE exact baseline signal set
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
    m1 = features["M1"]

    # -------------------------------------------------------------
    # M5 current production confirmation
    # -------------------------------------------------------------

    m5_confirmed = bool(
        setup.m5_valid
    )

    # -------------------------------------------------------------
    # M1 individual conditions
    # -------------------------------------------------------------

    m1_momentum = False
    m1_ema = False
    m1_location = False

    if m1 is not None:

        if (
            m1.momentum is not None
        ):
            if direction == "LONG":
                m1_momentum = (
                    m1.momentum > 0
                )
            elif direction == "SHORT":
                m1_momentum = (
                    m1.momentum < 0
                )

        if (
            m1.close is not None
            and m1.ema_fast is not None
        ):
            if direction == "LONG":
                m1_ema = (
                    m1.close
                    > m1.ema_fast
                )
            elif direction == "SHORT":
                m1_ema = (
                    m1.close
                    < m1.ema_fast
                )

        if (
            m1.close_location is not None
        ):
            if direction == "LONG":
                m1_location = (
                    m1.close_location
                    >= 0.60
                )
            elif direction == "SHORT":
                m1_location = (
                    m1.close_location
                    <= 0.40
                )

    m1_all = (
        m1_momentum
        and m1_ema
        and m1_location
    )

    records.append(
        {
            "decision_time": decision_time,
            "direction": direction,
            "score": decision.score,
            "m5_confirmed": m5_confirmed,
            "m1_momentum": m1_momentum,
            "m1_ema": m1_ema,
            "m1_location": m1_location,
            "m1_all": m1_all,
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
# Attach diagnostics to actual trades
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


print()
print("=" * 120)
print("M1 INCREMENTAL VALUE ANALYSIS")
print("=" * 120)

print()
print(
    f"Baseline trades : {len(trades)}"
)

print(
    f"M5-confirmed    : "
    f"{int(trades['m5_confirmed'].sum())}"
)

print(
    f"M5-not confirmed: "
    f"{int((~trades['m5_confirmed']).sum())}"
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

    pf = (
        gross_win / gross_loss
        if gross_loss
        else float("inf")
    )

    return {
        "n": len(p),
        "net": float(p.sum()),
        "win": float(
            (p > 0).mean() * 100
        ),
        "pf": pf,
        "avg": float(p.mean()),
    }


def fmt_pf(value):

    if np.isnan(value):
        return "N/A"

    if np.isinf(value):
        return "INF"

    return f"{value:.3f}"


# ---------------------------------------------------------------------
# M1 effect after M5
# ---------------------------------------------------------------------

m5_group = trades[
    trades["m5_confirmed"]
].copy()

no_m5_group = trades[
    ~trades["m5_confirmed"]
].copy()


print()
print("=" * 120)
print("M1 INSIDE M5-CONFIRMED TRADES")
print("=" * 120)

for name, mask in [

    (
        "M1_ALL",
        m5_group["m1_all"],
    ),

    (
        "M1_NOT_ALL",
        ~m5_group["m1_all"],
    ),

]:

    r = stats(
        m5_group.loc[mask]
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
# Each M1 condition after M5
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("INDIVIDUAL M1 CONDITIONS INSIDE M5")
print("=" * 120)

for name, column in [
    (
        "M1_MOMENTUM",
        "m1_momentum",
    ),
    (
        "M1_EMA20",
        "m1_ema",
    ),
    (
        "M1_LOCATION",
        "m1_location",
    ),
]:

    yes = stats(
        m5_group[
            m5_group[column]
        ]
    )

    no = stats(
        m5_group[
            ~m5_group[column]
        ]
    )

    print()
    print(name)

    print(
        f"  YES "
        f"trades={yes['n']:4d}  "
        f"net=${yes['net']:9.2f}  "
        f"win={yes['win']:6.2f}%  "
        f"PF={fmt_pf(yes['pf'])}"
    )

    print(
        f"  NO  "
        f"trades={no['n']:4d}  "
        f"net=${no['net']:9.2f}  "
        f"win={no['win']:6.2f}%  "
        f"PF={fmt_pf(no['pf'])}"
    )


# ---------------------------------------------------------------------
# All M1 combinations inside M5
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("M1 CONDITION COMBINATIONS INSIDE M5")
print("=" * 120)

combos = [
    (
        "MOM_ONLY",
        m5_group["m1_momentum"]
        & ~m5_group["m1_ema"]
        & ~m5_group["m1_location"],
    ),

    (
        "EMA_ONLY",
        ~m5_group["m1_momentum"]
        & m5_group["m1_ema"]
        & ~m5_group["m1_location"],
    ),

    (
        "LOCATION_ONLY",
        ~m5_group["m1_momentum"]
        & ~m5_group["m1_ema"]
        & m5_group["m1_location"],
    ),

    (
        "MOM+EMA",
        m5_group["m1_momentum"]
        & m5_group["m1_ema"]
        & ~m5_group["m1_location"],
    ),

    (
        "MOM+LOCATION",
        m5_group["m1_momentum"]
        & ~m5_group["m1_ema"]
        & m5_group["m1_location"],
    ),

    (
        "EMA+LOCATION",
        ~m5_group["m1_momentum"]
        & m5_group["m1_ema"]
        & m5_group["m1_location"],
    ),

    (
        "ALL_THREE",
        m5_group["m1_momentum"]
        & m5_group["m1_ema"]
        & m5_group["m1_location"],
    ),

    (
        "NONE",
        ~m5_group["m1_momentum"]
        & ~m5_group["m1_ema"]
        & ~m5_group["m1_location"],
    ),
]


for name, mask in combos:

    r = stats(
        m5_group.loc[mask]
    )

    if r["n"] == 0:
        continue

    print(
        f"{name:18s} "
        f"trades={r['n']:4d}  "
        f"net=${r['net']:9.2f}  "
        f"win={r['win']:6.2f}%  "
        f"PF={fmt_pf(r['pf']):>7s}  "
        f"avg=${r['avg']:7.2f}"
    )


# ---------------------------------------------------------------------
# Compare M5-only against M5+M1
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("M5 ONLY VS M5 + M1")
print("=" * 120)

m5_only = m5_group[
    ~m5_group["m1_all"]
]

m5_plus_m1 = m5_group[
    m5_group["m1_all"]
]

for name, frame in [
    (
        "M5_ONLY",
        m5_only,
    ),
    (
        "M5_PLUS_M1",
        m5_plus_m1,
    ),
]:

    r = stats(frame)

    print(
        f"{name:15s} "
        f"trades={r['n']:4d}  "
        f"net=${r['net']:9.2f}  "
        f"win={r['win']:6.2f}%  "
        f"PF={fmt_pf(r['pf']):>7s}  "
        f"avg=${r['avg']:7.2f}"
    )


# ---------------------------------------------------------------------
# M1 by year, only where M5 confirms
# ---------------------------------------------------------------------

m5_group["year"] = (
    m5_group["entry_time"]
    .dt.year
)

print()
print("=" * 120)
print("M1 ALL CONDITION BY YEAR, INSIDE M5")
print("=" * 120)

for year, group in m5_group.groupby(
    "year"
):

    print()
    print(f"YEAR {year}")

    for name, mask in [
        (
            "M1_ALL",
            group["m1_all"],
        ),
        (
            "M1_NOT_ALL",
            ~group["m1_all"],
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
# Premium + M5 + M1
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("PREMIUM + M5 + M1")
print("=" * 120)

premium_m5 = m5_group[
    m5_group["score"] >= 85
]

if premium_m5.empty:

    print("No premium M5 trades.")

else:

    for name, mask in [
        (
            "M1_ALL",
            premium_m5["m1_all"],
        ),
        (
            "M1_NOT_ALL",
            ~premium_m5["m1_all"],
        ),
    ]:

        r = stats(
            premium_m5.loc[mask]
        )

        print(
            f"{name:15s} "
            f"trades={r['n']:4d}  "
            f"net=${r['net']:9.2f}  "
            f"win={r['win']:6.2f}%  "
            f"PF={fmt_pf(r['pf'])}"
        )


# ---------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------

all_p = trades[
    "pnl"
].dropna()

print()
print("=" * 120)
print("BASELINE CONTROL")
print("=" * 120)

print(
    f"Trades   : {len(all_p)}"
)

print(
    f"Net P/L  : "
    f"${float(all_p.sum()):.2f}"
)

print(
    f"Win rate : "
    f"{float((all_p > 0).mean() * 100):.2f}%"
)

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
    float(wins.sum()) / gross_loss
    if gross_loss
    else float("inf")
)

print(
    f"PF       : {pf:.3f}"
)

print()
print("=" * 120)
print("IMPORTANT")
print("=" * 120)

print("""
This is a diagnostic experiment.

The baseline trades are generated exactly once.

The analysis then asks:

    When M5 already confirms the trade,
    does M1 add useful information?

M1 is decomposed into:

    momentum
    close vs EMA20
    close-location value

The current production M1 trigger requires all three.

No production files were modified.
No Git commits were created.
""")
