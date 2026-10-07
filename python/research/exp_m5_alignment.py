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

# ---------------------------------------------------------------------
# Prepare aligned closed candles for every timeframe
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
# Variants
# ---------------------------------------------------------------------

variants = {
    "BASELINE": np.zeros(len(h1), dtype=bool),
    "M5_EMA20_50": np.zeros(len(h1), dtype=bool),
    "M5_PLUS_H1": np.zeros(len(h1), dtype=bool),
    "M5_PLUS_M15": np.zeros(len(h1), dtype=bool),
    "M5_H1_M15": np.zeros(len(h1), dtype=bool),
}

short_variants = {
    name: np.zeros(len(h1), dtype=bool)
    for name in variants
}


counts = {
    name: 0
    for name in variants
}


def trend_from_features(features):
    """
    Raw EMA20 / EMA50 trend.

    LONG  = EMA20 > EMA50
    SHORT = EMA20 < EMA50
    """
    if features is None:
        return None

    if features.ema_fast is None or features.ema_slow is None:
        return None

    if features.ema_fast > features.ema_slow:
        return "LONG"

    if features.ema_fast < features.ema_slow:
        return "SHORT"

    return None


# ---------------------------------------------------------------------
# Reconstruct exact decision pipeline
# ---------------------------------------------------------------------

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

    if direction not in {"LONG", "SHORT"}:
        continue

    counts["BASELINE"] += 1

    # ---------------------------------------------------------------
    # Raw EMA20/EMA50 trends
    # ---------------------------------------------------------------

    h1_trend = trend_from_features(features["H1"])
    m15_trend = trend_from_features(features["M15"])
    m5_trend = trend_from_features(features["M5"])

    # ---------------------------------------------------------------
    # BASELINE
    # ---------------------------------------------------------------

    if direction == "LONG":
        variants["BASELINE"][i] = True
    else:
        short_variants["BASELINE"][i] = True

    # ---------------------------------------------------------------
    # M5 EMA20/EMA50
    # ---------------------------------------------------------------

    m5_ok = m5_trend == direction

    if m5_ok:
        counts["M5_EMA20_50"] += 1

        if direction == "LONG":
            variants["M5_EMA20_50"][i] = True
        else:
            short_variants["M5_EMA20_50"][i] = True

    # ---------------------------------------------------------------
    # M5 + H1 agreement
    # ---------------------------------------------------------------

    m5_h1_ok = (
        m5_trend == direction
        and h1_trend == direction
    )

    if m5_h1_ok:
        counts["M5_PLUS_H1"] += 1

        if direction == "LONG":
            variants["M5_PLUS_H1"][i] = True
        else:
            short_variants["M5_PLUS_H1"][i] = True

    # ---------------------------------------------------------------
    # M5 + M15 agreement
    # ---------------------------------------------------------------

    m5_m15_ok = (
        m5_trend == direction
        and m15_trend == direction
    )

    if m5_m15_ok:
        counts["M5_PLUS_M15"] += 1

        if direction == "LONG":
            variants["M5_PLUS_M15"][i] = True
        else:
            short_variants["M5_PLUS_M15"][i] = True

    # ---------------------------------------------------------------
    # M5 + H1 + M15 agreement
    # ---------------------------------------------------------------

    full_alignment = (
        m5_trend == direction
        and h1_trend == direction
        and m15_trend == direction
    )

    if full_alignment:
        counts["M5_H1_M15"] += 1

        if direction == "LONG":
            variants["M5_H1_M15"][i] = True
        else:
            short_variants["M5_H1_M15"][i] = True


# ---------------------------------------------------------------------
# Backtest helper
# ---------------------------------------------------------------------

def run_backtest(name, long_arr, short_arr):

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
    ).run()

    trades = pd.DataFrame(result.trades)

    final_equity = float(result.equity.iloc[-1])
    pnl = final_equity - result.initial_balance

    if trades.empty:
        return {
            "variant": name,
            "signals": int(long_arr.sum() + short_arr.sum()),
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
    max_dd = float(drawdown.max())

    return {
        "variant": name,
        "signals": int(long_arr.sum() + short_arr.sum()),
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
        "max_dd": max_dd,
        "halted": result.halted,
    }


# ---------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------

results = []

for name in variants:
    results.append(
        run_backtest(
            name,
            variants[name],
            short_variants[name],
        )
    )


# ---------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------

print()
print("=" * 90)
print("M5 EMA20/EMA50 ALIGNMENT EXPERIMENT")
print("=" * 90)

print()
print("SIGNAL SURVIVAL")
print("-" * 90)

for name in variants:
    print(
        f"{name:20s} : "
        f"{counts[name]:5d} signals"
    )

print()
print("=" * 90)
print("BACKTEST RESULTS")
print("=" * 90)

for r in results:
    print(
        f"{r['variant']:20s} "
        f"signals={r['signals']:5d}  "
        f"trades={r['trades']:4d}  "
        f"net=${r['net']:9.2f}  "
        f"return={r['return']:7.2f}%  "
        f"win={r['win_rate']:6.2f}%  "
        f"PF={r['pf']:6.3f}  "
        f"DD=${r['max_dd']:8.2f}  "
        f"halted={r['halted']}"
    )

print()
print("=" * 90)
print("INTERPRETATION")
print("=" * 90)

print("""
BASELINE
  Existing executable MTF strategy.

M5_EMA20_50
  Requires M5 EMA20 > EMA50 for LONG
  or M5 EMA20 < EMA50 for SHORT.

M5_PLUS_H1
  Requires M5 EMA20/EMA50 alignment
  AND H1 EMA20/EMA50 agreement.

M5_PLUS_M15
  Requires M5 EMA20/EMA50 alignment
  AND M15 EMA20/EMA50 agreement.

M5_H1_M15
  Requires all three:
      H1 + M15 + M5
  to agree with the trade direction.

No production files were modified.
No Git commits were created.
""")
