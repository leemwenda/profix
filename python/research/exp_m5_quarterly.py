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
# Trend helper
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
# Build exact baseline + M5 EMA20/EMA50 signals
# ---------------------------------------------------------------------

def build_signals():

    baseline_long = np.zeros(len(h1), dtype=bool)
    baseline_short = np.zeros(len(h1), dtype=bool)

    m5_long = np.zeros(len(h1), dtype=bool)
    m5_short = np.zeros(len(h1), dtype=bool)

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

        # -------------------------------------------------------------
        # BASELINE
        # -------------------------------------------------------------

        if direction == "LONG":
            baseline_long[i] = True
        else:
            baseline_short[i] = True

        # -------------------------------------------------------------
        # M5 EMA20/EMA50 confirmation
        # -------------------------------------------------------------

        m5_trend = trend_from_features(features["M5"])

        if m5_trend == direction:

            if direction == "LONG":
                m5_long[i] = True
            else:
                m5_short[i] = True

    return (
        baseline_long,
        baseline_short,
        m5_long,
        m5_short,
    )


# ---------------------------------------------------------------------
# Backtest helper
# ---------------------------------------------------------------------

def run_backtest(
    name,
    long_arr,
    short_arr,
    start_time,
    end_time,
):

    # decision_times is already a NumPy-compatible datetime index,
    # so this produces a NumPy boolean mask directly.
    mask = (
        (decision_times >= start_time)
        & (decision_times < end_time)
    )

    long_period = long_arr & mask
    short_period = short_arr & mask

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
                long_period,
                short_period,
            )
        },
    ).run()

    trades = pd.DataFrame(result.trades)

    final_equity = float(
        result.equity.iloc[-1]
    )

    pnl = final_equity - result.initial_balance

    if trades.empty:
        return {
            "variant": name,
            "signals": int(
                long_period.sum()
                + short_period.sum()
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

    max_dd = float(drawdown.max())

    return {
        "variant": name,
        "signals": int(
            long_period.sum()
            + short_period.sum()
        ),
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
# Quarterly periods
#
# M5 data begins on 2025-04-30.
# Therefore 2025 Q2 starts at the first available M5 data.
# ---------------------------------------------------------------------

periods = [
    (
        "2025_Q2",
        pd.Timestamp("2025-04-30", tz="UTC"),
        pd.Timestamp("2025-07-01", tz="UTC"),
    ),
    (
        "2025_Q3",
        pd.Timestamp("2025-07-01", tz="UTC"),
        pd.Timestamp("2025-10-01", tz="UTC"),
    ),
    (
        "2025_Q4",
        pd.Timestamp("2025-10-01", tz="UTC"),
        pd.Timestamp("2026-01-01", tz="UTC"),
    ),
    (
        "2026_Q1",
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-04-01", tz="UTC"),
    ),
    (
        "2026_Q2",
        pd.Timestamp("2026-04-01", tz="UTC"),
        pd.Timestamp("2026-07-01", tz="UTC"),
    ),
    (
        "2026_Q3",
        pd.Timestamp("2026-07-01", tz="UTC"),
        pd.Timestamp("2026-10-01", tz="UTC"),
    ),
    (
        "2026_Q4_YTD",
        pd.Timestamp("2026-10-01", tz="UTC"),
        pd.Timestamp("2027-01-01", tz="UTC"),
    ),
]


# ---------------------------------------------------------------------
# Build signals once
# ---------------------------------------------------------------------

(
    baseline_long,
    baseline_short,
    m5_long,
    m5_short,
) = build_signals()


# ---------------------------------------------------------------------
# Run quarterly experiments
# ---------------------------------------------------------------------

all_results = []

for period_name, start_time, end_time in periods:

    baseline = run_backtest(
        "BASELINE",
        baseline_long,
        baseline_short,
        start_time,
        end_time,
    )

    m5 = run_backtest(
        "M5_EMA20_50",
        m5_long,
        m5_short,
        start_time,
        end_time,
    )

    all_results.append(
        (
            period_name,
            baseline,
            m5,
        )
    )


# ---------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("QUARTERLY M5 EMA20/EMA50 STABILITY EXPERIMENT")
print("=" * 110)

print()
print(
    f"{'PERIOD':15s} "
    f"{'VARIANT':15s} "
    f"{'SIGNALS':>8s} "
    f"{'TRADES':>7s} "
    f"{'NET':>10s} "
    f"{'RETURN':>9s} "
    f"{'WIN':>8s} "
    f"{'PF':>7s} "
    f"{'DD':>10s}"
)

print("-" * 110)

for period_name, baseline, m5 in all_results:

    for r in (baseline, m5):

        pf = r["pf"]

        if np.isnan(pf):
            pf_text = "N/A"
        elif np.isinf(pf):
            pf_text = "INF"
        else:
            pf_text = f"{pf:.3f}"

        print(
            f"{period_name:15s} "
            f"{r['variant']:15s} "
            f"{r['signals']:8d} "
            f"{r['trades']:7d} "
            f"${r['net']:9.2f} "
            f"{r['return']:8.2f}% "
            f"{r['win_rate']:7.2f}% "
            f"{pf_text:>7s} "
            f"${r['max_dd']:9.2f}"
        )

    print("-" * 110)


# ---------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("M5 IMPROVEMENT BY QUARTER")
print("=" * 110)

for period_name, baseline, m5 in all_results:

    return_delta = (
        m5["return"]
        - baseline["return"]
    )

    if (
        np.isfinite(m5["pf"])
        and np.isfinite(baseline["pf"])
    ):
        pf_delta = m5["pf"] - baseline["pf"]
        pf_text = f"{pf_delta:+7.3f}"
    else:
        pf_text = "    N/A"

    print(
        f"{period_name:15s} "
        f"return_delta={return_delta:+7.2f}%  "
        f"PF_delta={pf_text}  "
        f"M5_signals={m5['signals']:4d}"
    )


print()
print("=" * 110)
print("IMPORTANT")
print("=" * 110)

print("""
2025 Q2 is the first period that can be evaluated with M5 data.

BASELINE
    Existing executable MTF strategy.

M5_EMA20_50
    Existing executable MTF strategy
    +
    M5 EMA20 > EMA50 for LONG
    M5 EMA20 < EMA50 for SHORT.

No production files were modified.
No Git commits were created.
""")
