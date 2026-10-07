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
# Build exact executable baseline + M5 confirmation
# ---------------------------------------------------------------------

baseline_long = np.zeros(len(h1), dtype=bool)
baseline_short = np.zeros(len(h1), dtype=bool)

m5_long = np.zeros(len(h1), dtype=bool)
m5_short = np.zeros(len(h1), dtype=bool)

m15_ratio = np.full(len(h1), np.nan)


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
    # M5 EMA20 / EMA50
    # -------------------------------------------------------------

    m5_trend = trend_from_features(features["M5"])

    if m5_trend == direction:

        if direction == "LONG":
            m5_long[i] = True
        else:
            m5_short[i] = True

    # -------------------------------------------------------------
    # Independent M15 momentum strength
    # -------------------------------------------------------------

    m15 = features["M15"]

    if (
        m15 is not None
        and m15.momentum is not None
        and m15.atr is not None
        and m15.atr > 0
    ):
        m15_ratio[i] = (
            abs(float(m15.momentum))
            / float(m15.atr)
        )


# ---------------------------------------------------------------------
# Quarterly validation periods
#
# We already tested 2025 Q2/Q3 extensively.
# This OOS experiment begins with 2025 Q4.
#
# Thresholds for each quarter are learned ONLY from
# all previous available data.
# ---------------------------------------------------------------------

periods = [
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

TRAIN_START = pd.Timestamp("2025-04-30", tz="UTC")


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

    period_mask = (
        (decision_times >= start_time)
        & (decision_times < end_time)
    )

    long_period = long_arr & period_mask
    short_period = short_arr & period_mask

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

    signals = int(
        long_period.sum()
        + short_period.sum()
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
        "max_dd": float(drawdown.max()),
    }


# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------

all_results = []


# ---------------------------------------------------------------------
# Expanding walk-forward
# ---------------------------------------------------------------------

for period_name, validation_start, validation_end in periods:

    # ---------------------------------------------------------------
    # TRAINING DATA
    #
    # IMPORTANT:
    # validation_start is NOT included.
    # ---------------------------------------------------------------

    train_mask = (
        (decision_times >= TRAIN_START)
        & (decision_times < validation_start)
        & (baseline_long | baseline_short)
        & np.isfinite(m15_ratio)
    )

    train_ratios = m15_ratio[train_mask]

    q1 = float(np.quantile(train_ratios, 1 / 3))
    q2 = float(np.quantile(train_ratios, 2 / 3))

    # ---------------------------------------------------------------
    # Validation regime masks using FROZEN training thresholds
    # ---------------------------------------------------------------

    validation_window = (
        (decision_times >= validation_start)
        & (decision_times < validation_end)
    )

    low_mask = (
        validation_window
        & np.isfinite(m15_ratio)
        & (m15_ratio <= q1)
    )

    mid_mask = (
        validation_window
        & np.isfinite(m15_ratio)
        & (m15_ratio > q1)
        & (m15_ratio <= q2)
    )

    high_mask = (
        validation_window
        & np.isfinite(m15_ratio)
        & (m15_ratio > q2)
    )

    # ---------------------------------------------------------------
    # BASELINE
    # ---------------------------------------------------------------

    base = run_backtest(
        "BASELINE",
        baseline_long,
        baseline_short,
        validation_start,
        validation_end,
    )

    # ---------------------------------------------------------------
    # M5 ALWAYS
    # ---------------------------------------------------------------

    m5_always = run_backtest(
        "M5_ALWAYS",
        m5_long,
        m5_short,
        validation_start,
        validation_end,
    )

    # ---------------------------------------------------------------
    # M5 HIGH ONLY
    # ---------------------------------------------------------------

    high_long = baseline_long.copy()
    high_short = baseline_short.copy()

    high_long[high_mask] = m5_long[high_mask]
    high_short[high_mask] = m5_short[high_mask]

    m5_high = run_backtest(
        "M5_HIGH_ONLY",
        high_long,
        high_short,
        validation_start,
        validation_end,
    )

    # ---------------------------------------------------------------
    # M5 MID + HIGH
    # ---------------------------------------------------------------

    mid_high_mask = mid_mask | high_mask

    mid_high_long = baseline_long.copy()
    mid_high_short = baseline_short.copy()

    mid_high_long[mid_high_mask] = (
        m5_long[mid_high_mask]
    )

    mid_high_short[mid_high_mask] = (
        m5_short[mid_high_mask]
    )

    m5_mid_high = run_backtest(
        "M5_MID_HIGH",
        mid_high_long,
        mid_high_short,
        validation_start,
        validation_end,
    )

    all_results.append(
        {
            "period": period_name,
            "q1": q1,
            "q2": q2,
            "train_signals": int(train_mask.sum()),
            "baseline": base,
            "m5_always": m5_always,
            "m5_high": m5_high,
            "m5_mid_high": m5_mid_high,
        }
    )


# ---------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("OUT-OF-SAMPLE M5 CONDITIONAL WALK-FORWARD")
print("=" * 125)

print()
print(
    f"{'PERIOD':13s} "
    f"{'VARIANT':15s} "
    f"{'SIGNALS':>8s} "
    f"{'TRADES':>7s} "
    f"{'NET':>10s} "
    f"{'RETURN':>9s} "
    f"{'WIN':>8s} "
    f"{'PF':>7s} "
    f"{'DD':>10s}"
)

print("-" * 125)

for item in all_results:

    for r in (
        item["baseline"],
        item["m5_always"],
        item["m5_high"],
        item["m5_mid_high"],
    ):

        pf = r["pf"]

        if np.isnan(pf):
            pf_text = "N/A"
        elif np.isinf(pf):
            pf_text = "INF"
        else:
            pf_text = f"{pf:.3f}"

        print(
            f"{item['period']:13s} "
            f"{r['variant']:15s} "
            f"{r['signals']:8d} "
            f"{r['trades']:7d} "
            f"${r['net']:9.2f} "
            f"{r['return']:8.2f}% "
            f"{r['win_rate']:7.2f}% "
            f"{pf_text:>7s} "
            f"${r['max_dd']:9.2f}"
        )

    print(
        f"  TRAIN signals={item['train_signals']}  "
        f"Q1={item['q1']:.6f}  "
        f"Q2={item['q2']:.6f}"
    )

    print("-" * 125)


# ---------------------------------------------------------------------
# Out-of-sample deltas
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("OUT-OF-SAMPLE DELTAS VS BASELINE")
print("=" * 125)

for item in all_results:

    base = item["baseline"]

    print()
    print(item["period"])

    for label, candidate in (
        ("M5_ALWAYS", item["m5_always"]),
        ("M5_HIGH_ONLY", item["m5_high"]),
        ("M5_MID_HIGH", item["m5_mid_high"]),
    ):

        return_delta = (
            candidate["return"]
            - base["return"]
        )

        if (
            np.isfinite(candidate["pf"])
            and np.isfinite(base["pf"])
        ):
            pf_delta = (
                candidate["pf"]
                - base["pf"]
            )
            pf_text = f"{pf_delta:+.3f}"
        else:
            pf_text = "N/A"

        print(
            f"  {label:15s} "
            f"return_delta={return_delta:+.2f}%  "
            f"PF_delta={pf_text}"
        )


# ---------------------------------------------------------------------
# Aggregate OOS scorecard
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("OOS SCORECARD")
print("=" * 125)

variants = {
    "M5_ALWAYS": "m5_always",
    "M5_HIGH_ONLY": "m5_high",
    "M5_MID_HIGH": "m5_mid_high",
}

for name, key in variants.items():

    wins = 0
    losses = 0
    same = 0

    compounded = 1.0
    total_net = 0.0

    for item in all_results:

        base = item["baseline"]
        candidate = item[key]

        delta = (
            candidate["return"]
            - base["return"]
        )

        if delta > 0.0001:
            wins += 1
        elif delta < -0.0001:
            losses += 1
        else:
            same += 1

        compounded *= (
            1.0
            + candidate["return"] / 100.0
        )

        total_net += candidate["net"]

    compounded_return = (
        compounded - 1.0
    ) * 100

    print(
        f"{name:15s} "
        f"better={wins}  "
        f"worse={losses}  "
        f"same={same}  "
        f"compounded_OOS={compounded_return:+.2f}%  "
        f"sum_net=${total_net:+.2f}"
    )


print()
print("=" * 125)
print("INTERPRETATION")
print("=" * 125)

print("""
This is a genuine out-of-sample experiment.

For every validation quarter:

1. M15 momentum thresholds are calculated only from earlier data.
2. Those thresholds are frozen.
3. The current quarter is then tested without using its results
   to define the thresholds.

BASELINE
    Existing executable MTF strategy.

M5_ALWAYS
    M5 EMA20/EMA50 required for every executable signal.

M5_HIGH_ONLY
    M5 EMA20/EMA50 required only when M15 momentum is above
    the training-period upper tercile.

M5_MID_HIGH
    M5 EMA20/EMA50 required when M15 momentum is above the
    training-period lower tercile.

No production files were modified.
No Git commits were created.
""")
