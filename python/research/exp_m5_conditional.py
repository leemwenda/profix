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
# Prepare aligned closed candles
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

def trend_from_features(features):
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
# Build baseline, M5 filter and independent M15 momentum strength
# ---------------------------------------------------------------------

baseline_long = np.zeros(len(h1), dtype=bool)
baseline_short = np.zeros(len(h1), dtype=bool)

m5_long = np.zeros(len(h1), dtype=bool)
m5_short = np.zeros(len(h1), dtype=bool)

m15_momentum_ratio = np.full(len(h1), np.nan)

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

    executable = decision.executable and market_ok

    if not executable:
        continue

    direction = decision.direction

    if direction not in {"LONG", "SHORT"}:
        continue

    executable_count += 1

    # -------------------------------------------------------------
    # BASELINE
    # -------------------------------------------------------------

    if direction == "LONG":
        baseline_long[i] = True
    else:
        baseline_short[i] = True

    # -------------------------------------------------------------
    # M5 EMA20 / EMA50 filter
    # -------------------------------------------------------------

    m5_trend = trend_from_features(features["M5"])

    if m5_trend == direction:

        if direction == "LONG":
            m5_long[i] = True
        else:
            m5_short[i] = True

    # -------------------------------------------------------------
    # M15 momentum strength
    #
    # IMPORTANT:
    # This uses M15 only.
    # It does NOT use M5, so the conditioning variable is
    # independent of the M5 filter being tested.
    #
    # ratio = absolute momentum / ATR
    # -------------------------------------------------------------

    m15 = features["M15"]

    if (
        m15 is not None
        and m15.momentum is not None
        and m15.atr is not None
        and m15.atr > 0
    ):
        m15_momentum_ratio[i] = (
            abs(float(m15.momentum))
            / float(m15.atr)
        )


# ---------------------------------------------------------------------
# Determine LOW / MID / HIGH M15 momentum groups
# ---------------------------------------------------------------------

cut = decision_times < pd.Timestamp("2025-05-02", tz="UTC")
baseline_long[cut] = False
baseline_short[cut] = False
m5_long[cut] = False
m5_short[cut] = False

valid_ratios = m15_momentum_ratio[
    np.isfinite(m15_momentum_ratio)
    & (baseline_long | baseline_short)
]

q1 = float(np.quantile(valid_ratios, 1 / 3))
q2 = float(np.quantile(valid_ratios, 2 / 3))


low_mask = (
    np.isfinite(m15_momentum_ratio)
    & (m15_momentum_ratio <= q1)
)

mid_mask = (
    np.isfinite(m15_momentum_ratio)
    & (m15_momentum_ratio > q1)
    & (m15_momentum_ratio <= q2)
)

high_mask = (
    np.isfinite(m15_momentum_ratio)
    & (m15_momentum_ratio > q2)
)


# ---------------------------------------------------------------------
# Utility for masking signals
# ---------------------------------------------------------------------

def apply_mask(long_arr, short_arr, mask):
    return (
        long_arr & mask,
        short_arr & mask,
    )


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

    final_equity = float(
        result.equity.iloc[-1]
    )

    pnl = final_equity - result.initial_balance

    signals = int(
        long_arr.sum()
        + short_arr.sum()
    )

    if trades.empty:
        return {
            "variant": name,
            "signals": signals,
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
        "signals": signals,
        "trades": len(trades),
        "net": pnl,
        "return": (
            final_equity / result.initial_balance - 1
        ) * 100,
        "win_rate": float(
            (p > 0).mean() * 100
        ),
        "pf": (
            gross_win / gross_loss
            if gross_loss
            else float("inf")
        ),
        "max_dd": max_dd,
        "halted": result.halted,
    }


# ---------------------------------------------------------------------
# Group-specific baseline and M5 results
# ---------------------------------------------------------------------

group_masks = {
    "LOW_M15_MOM": low_mask,
    "MID_M15_MOM": mid_mask,
    "HIGH_M15_MOM": high_mask,
}

results = []


# ---------------------------------------------------------------------
# Overall comparison
# ---------------------------------------------------------------------

results.append(
    run_backtest(
        "BASELINE",
        baseline_long,
        baseline_short,
    )
)

results.append(
    run_backtest(
        "M5_ALWAYS",
        m5_long,
        m5_short,
    )
)


# ---------------------------------------------------------------------
# Compare M5 inside each M15 momentum group
# ---------------------------------------------------------------------

for group_name, mask in group_masks.items():

    base_long, base_short = apply_mask(
        baseline_long,
        baseline_short,
        mask,
    )

    m5_group_long, m5_group_short = apply_mask(
        m5_long,
        m5_short,
        mask,
    )

    results.append(
        run_backtest(
            f"{group_name}_BASE",
            base_long,
            base_short,
        )
    )

    results.append(
        run_backtest(
            f"{group_name}_M5",
            m5_group_long,
            m5_group_short,
        )
    )


# ---------------------------------------------------------------------
# Conditional strategies
#
# These are the important candidates:
#
# M5_LOW_ONLY:
#   Require M5 confirmation only when M15 momentum is LOW.
#
# M5_MID_ONLY:
#   Require M5 confirmation only when M15 momentum is MID.
#
# M5_HIGH_ONLY:
#   Require M5 confirmation only when M15 momentum is HIGH.
#
# Outside the selected group, the baseline signal is retained.
# ---------------------------------------------------------------------

conditional_variants = {}


for selected_name, selected_mask in group_masks.items():

    long_conditional = baseline_long.copy()
    short_conditional = baseline_short.copy()

    # Remove baseline signals from the selected group.
    long_conditional[selected_mask] = False
    short_conditional[selected_mask] = False

    # Put M5-confirmed signals back into the selected group.
    long_conditional[selected_mask] = (
        m5_long[selected_mask]
    )

    short_conditional[selected_mask] = (
        m5_short[selected_mask]
    )

    variant_name = f"M5_{selected_name}_ONLY"

    conditional_variants[variant_name] = (
        long_conditional,
        short_conditional,
    )

    results.append(
        run_backtest(
            variant_name,
            long_conditional,
            short_conditional,
        )
    )


# ---------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("M5 CONDITIONAL FILTER EXPERIMENT")
print("=" * 110)

print()
print(f"Executable baseline signals : {executable_count}")
print(f"M15 momentum Q1              : {q1:.4f}")
print(f"M15 momentum Q2              : {q2:.4f}")

print()
print(
    f"{'VARIANT':25s} "
    f"{'SIGNALS':>8s} "
    f"{'TRADES':>7s} "
    f"{'NET':>10s} "
    f"{'RETURN':>9s} "
    f"{'WIN':>8s} "
    f"{'PF':>7s} "
    f"{'DD':>10s}"
)

print("-" * 110)

for r in results:

    pf = r["pf"]

    if np.isnan(pf):
        pf_text = "N/A"
    elif np.isinf(pf):
        pf_text = "INF"
    else:
        pf_text = f"{pf:.3f}"

    print(
        f"{r['variant']:25s} "
        f"{r['signals']:8d} "
        f"{r['trades']:7d} "
        f"${r['net']:9.2f} "
        f"{r['return']:8.2f}% "
        f"{r['win_rate']:7.2f}% "
        f"{pf_text:>7s} "
        f"${r['max_dd']:9.2f}"
    )


# ---------------------------------------------------------------------
# Group comparison
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("M5 EFFECT INSIDE EACH M15 MOMENTUM GROUP")
print("=" * 110)

result_map = {
    r["variant"]: r
    for r in results
}

for group_name in group_masks:

    base = result_map[f"{group_name}_BASE"]
    m5 = result_map[f"{group_name}_M5"]

    return_delta = (
        m5["return"]
        - base["return"]
    )

    if (
        np.isfinite(m5["pf"])
        and np.isfinite(base["pf"])
    ):
        pf_delta = m5["pf"] - base["pf"]
        pf_text = f"{pf_delta:+.3f}"
    else:
        pf_text = "N/A"

    print()
    print(group_name)

    print(
        f"  BASELINE : "
        f"return={base['return']:+.2f}%  "
        f"PF={base['pf']:.3f}  "
        f"trades={base['trades']}"
    )

    print(
        f"  M5       : "
        f"return={m5['return']:+.2f}%  "
        f"PF={m5['pf']:.3f}  "
        f"trades={m5['trades']}"
    )

    print(
        f"  M5 delta : "
        f"return={return_delta:+.2f}%  "
        f"PF={pf_text}"
    )


print()
print("=" * 110)
print("CONDITIONAL STRATEGY COMPARISON")
print("=" * 110)

for name in conditional_variants:

    r = result_map[name]

    print(
        f"{name:25s} "
        f"return={r['return']:+.2f}%  "
        f"PF={r['pf']:.3f}  "
        f"trades={r['trades']}  "
        f"DD=${r['max_dd']:.2f}"
    )


print()
print("=" * 110)
print("INTERPRETATION")
print("=" * 110)

print("""
LOW_M15_MOM
    Bottom third of executable signals by
    absolute M15 momentum / M15 ATR.

MID_M15_MOM
    Middle third.

HIGH_M15_MOM
    Top third.

The conditional strategies keep the BASELINE behavior everywhere
except the selected momentum group, where M5 EMA20/EMA50 confirmation
is required.

This experiment is intended to discover WHERE M5 helps.

No production files were modified.
No Git commits were created.
""")
