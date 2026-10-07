from pathlib import Path

import numpy as np
import pandas as pd

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


ROOT = Path(".")
DATA_DIR = ROOT / "python/data/mtf"
CFG = load_config(ROOT / "config/xauusd.toml")
SYMBOL = "XAUUSD"

mtf = load_mtf(DATA_DIR, SYMBOL)
h1 = mtf.frame("H1")

decision_times = h1.index + pd.Timedelta(hours=1)

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


baseline_long = np.zeros(len(h1), dtype=bool)
baseline_short = np.zeros(len(h1), dtype=bool)

candidate_long = np.zeros(len(h1), dtype=bool)
candidate_short = np.zeros(len(h1), dtype=bool)


def trend(features):
    if features is None:
        return None

    if features.ema_fast is None or features.ema_slow is None:
        return None

    if features.ema_fast > features.ema_slow:
        return "LONG"

    if features.ema_fast < features.ema_slow:
        return "SHORT"

    return None


# ============================================================
# Reconstruct the exact production decision pipeline
# ============================================================

for i, decision_time in enumerate(decision_times):

    decision_time = pd.Timestamp(decision_time).tz_convert("UTC")

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

    executable = decision.executable and market_ok

    if not executable:
        continue

    direction = decision.direction

    if direction == "LONG":
        baseline_long[i] = True

    elif direction == "SHORT":
        baseline_short[i] = True

    else:
        continue

    # Candidate:
    # M5 EMA20/EMA50 must agree with the trade direction.
    m5_trend = trend(features["M5"])

    if m5_trend == direction:

        if direction == "LONG":
            candidate_long[i] = True

        elif direction == "SHORT":
            candidate_short[i] = True


# ============================================================
# Backtest one period
# ============================================================

def run_period(
    name,
    start,
    end,
    long_signals,
    short_signals,
):

    mask = (
        (h1.index >= start)
        & (h1.index < end)
    )

    period_h1 = h1.loc[mask]

    if period_h1.empty:
        return None

    positions = np.flatnonzero(mask)

    long_period = long_signals[positions]
    short_period = short_signals[positions]

    market = MarketData(
        {"XAUUSD": period_h1},
        htf_hours=CFG.htf_hours,
        htf_offset=CFG.htf_offset_hours,
    )

    result = BacktestEngine(
        market,
        CFG,
        signal_override={
            "XAUUSD": (
                long_period,
                short_period,
            )
        },
    ).run()

    trades = pd.DataFrame(result.trades)

    final_equity = float(result.equity.iloc[-1])
    pnl = final_equity - result.initial_balance

    if trades.empty:
        return {
            "name": name,
            "signals": int(
                long_period.sum() + short_period.sum()
            ),
            "trades": 0,
            "net": pnl,
            "return": 0.0,
            "win_rate": 0.0,
            "pf": float("nan"),
            "max_dd": 0.0,
            "halted": result.halted,
        }

    p = pd.to_numeric(
        trades["pnl"],
        errors="coerce",
    )

    wins = p[p > 0]
    losses = p[p < 0]

    gross_win = float(wins.sum())
    gross_loss = abs(float(losses.sum()))

    equity = pd.Series(result.equity)

    running_max = equity.cummax()
    drawdown = running_max - equity

    return {
        "name": name,
        "signals": int(
            long_period.sum() + short_period.sum()
        ),
        "trades": len(trades),
        "net": pnl,
        "return": (
            final_equity / result.initial_balance - 1
        ) * 100,
        "win_rate": float((p > 0).mean() * 100),
        "pf": (
            gross_win / gross_loss
            if gross_loss
            else float("inf")
        ),
        "max_dd": float(drawdown.max()),
        "halted": result.halted,
    }


# ============================================================
# Yearly walk-forward evaluation
# ============================================================

years = [
    ("2022", "2022-01-01", "2023-01-01"),
    ("2023", "2023-01-01", "2024-01-01"),
    ("2024", "2024-01-01", "2025-01-01"),
    ("2025", "2025-01-01", "2026-01-01"),
    ("2026_YTD", "2026-01-01", "2027-01-01"),
]


print()
print("=" * 110)
print("M5 EMA20/EMA50 WALK-FORWARD STABILITY TEST")
print("=" * 110)

all_results = []

for year, start, end in years:

    baseline = run_period(
        f"{year} BASELINE",
        start,
        end,
        baseline_long,
        baseline_short,
    )

    candidate = run_period(
        f"{year} M5_EMA20_50",
        start,
        end,
        candidate_long,
        candidate_short,
    )

    if baseline is None:
        continue

    print()
    print("=" * 110)
    print(year)
    print("=" * 110)

    for r in (baseline, candidate):

        print(
            f"{r['name']:22s} "
            f"signals={r['signals']:5d}  "
            f"trades={r['trades']:4d}  "
            f"net=${r['net']:9.2f}  "
            f"return={r['return']:8.2f}%  "
            f"win={r['win_rate']:6.2f}%  "
            f"PF={r['pf']:6.3f}  "
            f"DD=${r['max_dd']:8.2f}  "
            f"halted={r['halted']}"
        )

        all_results.append(r)


print()
print("=" * 110)
print("YEAR-BY-YEAR CANDIDATE CHECK")
print("=" * 110)

candidate_results = [
    r for r in all_results
    if "M5_EMA20_50" in r["name"]
]

profitable_years = sum(
    r["net"] > 0
    for r in candidate_results
)

losing_years = sum(
    r["net"] < 0
    for r in candidate_results
)

print(
    f"Profitable years : {profitable_years}"
)

print(
    f"Losing years     : {losing_years}"
)

print()
print("Candidate yearly results:")

for r in candidate_results:
    print(
        f"  {r['name']:20s} "
        f"return={r['return']:8.2f}%  "
        f"PF={r['pf']:6.3f}  "
        f"trades={r['trades']:4d}"
    )


print()
print("=" * 110)
print("IMPORTANT")
print("=" * 110)

print("""
This is a validation experiment only.

No production files were modified.
No strategy parameters were changed.
No Git commit was created.

The candidate is:

    H1 regime
        ↓
    M15 setup
        ↓
    M5 EMA20/EMA50 confirmation
        ↓
    EXECUTE

M1 is not required.
""")
