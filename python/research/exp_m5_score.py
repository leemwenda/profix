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

def clamp_score(value):
    return max(0, min(100, int(value)))


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


# ---------------------------------------------------------------------
# Store executable signals and scores
# ---------------------------------------------------------------------

baseline_long = np.zeros(len(h1), dtype=bool)
baseline_short = np.zeros(len(h1), dtype=bool)

current_score = np.full(len(h1), np.nan)
ema_score_5 = np.full(len(h1), np.nan)
ema_score_10 = np.full(len(h1), np.nan)

ema_alignment = np.zeros(len(h1), dtype=bool)

executable_count = 0


# ---------------------------------------------------------------------
# Exact current decision pipeline
# ---------------------------------------------------------------------

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

    # -------------------------------------------------------------
    # IMPORTANT:
    # assess_regime() already contains M5 trend information.
    # We therefore DO NOT alter regime scoring here.
    # We only replace the M5 component of setup scoring.
    # -------------------------------------------------------------

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
        baseline_long[i] = True
    else:
        baseline_short[i] = True

    # -------------------------------------------------------------
    # CURRENT SCORE
    #
    # Current setup:
    # M15 = 15
    # M5  = 10
    # M1  = 5
    #
    # We remove ONLY the current M5 contribution.
    # -------------------------------------------------------------

    current_m5_points = (
        10 if setup.m5_valid else 0
    )

    score_without_m5 = (
        decision.score
        - current_m5_points
    )

    current_score[i] = decision.score

    # -------------------------------------------------------------
    # NEW M5 EMA20/EMA50 alignment
    # -------------------------------------------------------------

    m5_trend = ema_trend(features["M5"])

    ema_ok = (
        m5_trend == direction
    )

    if ema_ok:
        ema_alignment[i] = True

    ema_bonus_5 = 5 if ema_ok else 0
    ema_bonus_10 = 10 if ema_ok else 0

    ema_score_5[i] = clamp_score(
        score_without_m5
        + ema_bonus_5
    )

    ema_score_10[i] = clamp_score(
        score_without_m5
        + ema_bonus_10
    )


# ---------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------

def run_backtest(
    name,
    long_arr,
    short_arr,
    start_time,
    end_time,
):

    window_mask = (
        (decision_times >= start_time)
        & (decision_times < end_time)
    )

    long_period = long_arr & window_mask
    short_period = short_arr & window_mask

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
        window=(start_time, end_time),
    ).run()

    trades = pd.DataFrame(result.trades)

    final_equity = float(
        result.equity.iloc[-1]
    )

    pnl = (
        final_equity
        - result.initial_balance
    )

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

    gross_win = float(
        wins.sum()
    )

    gross_loss = abs(
        float(losses.sum())
    )

    equity = pd.Series(
        result.equity
    )

    running_max = equity.cummax()

    drawdown = (
        running_max
        - equity
    )

    max_dd = float(
        drawdown.max()
    )

    return {
        "variant": name,
        "signals": signals,
        "trades": len(trades),
        "net": pnl,
        "return": (
            final_equity
            / result.initial_balance
            - 1
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
    }


# ---------------------------------------------------------------------
# Periods
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
# Score-based signal arrays
# ---------------------------------------------------------------------

variants = {}


for threshold in (65, 75, 85):

    # Current production scoring
    current_ok = (
        np.isfinite(current_score)
        & (current_score >= threshold)
    )

    # EMA20/EMA50 with +5 points
    ema5_ok = (
        np.isfinite(ema_score_5)
        & (ema_score_5 >= threshold)
    )

    # EMA20/EMA50 with +10 points
    ema10_ok = (
        np.isfinite(ema_score_10)
        & (ema_score_10 >= threshold)
    )

    variants[
        f"CURRENT_SCORE_{threshold}"
    ] = (
        baseline_long & current_ok,
        baseline_short & current_ok,
    )

    variants[
        f"EMA_SCORE_5_{threshold}"
    ] = (
        baseline_long & ema5_ok,
        baseline_short & ema5_ok,
    )

    variants[
        f"EMA_SCORE_10_{threshold}"
    ] = (
        baseline_long & ema10_ok,
        baseline_short & ema10_ok,
    )


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

results = []


for period_name, start_time, end_time in periods:

    for name, (
        long_arr,
        short_arr,
    ) in variants.items():

        r = run_backtest(
            name,
            long_arr,
            short_arr,
            start_time,
            end_time,
        )

        results.append(
            (period_name, r)
        )


# ---------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("M5 SOFT SCORE EXPERIMENT")
print("=" * 125)

print()
print(
    f"Executable baseline signals : "
    f"{executable_count}"
)

print(
    f"M5 EMA20/EMA50 aligned      : "
    f"{int(ema_alignment.sum())}"
)

print()
print(
    f"{'PERIOD':13s} "
    f"{'VARIANT':22s} "
    f"{'SIGNALS':>8s} "
    f"{'TRADES':>7s} "
    f"{'NET':>10s} "
    f"{'RETURN':>9s} "
    f"{'WIN':>8s} "
    f"{'PF':>7s} "
    f"{'DD':>10s}"
)

print("-" * 125)

for period_name, r in results:

    pf = r["pf"]

    if np.isnan(pf):
        pf_text = "N/A"
    elif np.isinf(pf):
        pf_text = "INF"
    else:
        pf_text = f"{pf:.3f}"

    print(
        f"{period_name:13s} "
        f"{r['variant']:22s} "
        f"{r['signals']:8d} "
        f"{r['trades']:7d} "
        f"${r['net']:9.2f} "
        f"{r['return']:8.2f}% "
        f"{r['win_rate']:7.2f}% "
        f"{pf_text:>7s} "
        f"${r['max_dd']:9.2f}"
    )


# ---------------------------------------------------------------------
# Aggregate scorecard
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("SCORE THRESHOLD SUMMARY")
print("=" * 125)

result_map = {}

for period_name, r in results:
    result_map[
        (period_name, r["variant"])
    ] = r


for threshold in (65, 75, 85):

    print()
    print(
        f"THRESHOLD {threshold}"
    )

    print(
        f"{'PERIOD':13s} "
        f"{'CURRENT':>12s} "
        f"{'EMA +5':>12s} "
        f"{'EMA +10':>12s}"
    )

    print("-" * 55)

    for (
        period_name,
        _start,
        _end,
    ) in periods:

        current = result_map[
            (
                period_name,
                f"CURRENT_SCORE_{threshold}",
            )
        ]

        ema5 = result_map[
            (
                period_name,
                f"EMA_SCORE_5_{threshold}",
            )
        ]

        ema10 = result_map[
            (
                period_name,
                f"EMA_SCORE_10_{threshold}",
            )
        ]

        print(
            f"{period_name:13s} "
            f"{current['return']:+7.2f}% "
            f"PF {current['pf']:.3f}   "
            f"{ema5['return']:+7.2f}% "
            f"PF {ema5['pf']:.3f}   "
            f"{ema10['return']:+7.2f}% "
            f"PF {ema10['pf']:.3f}"
        )


print()
print("=" * 125)
print("IMPORTANT")
print("=" * 125)

print("""
CURRENT_SCORE
    Existing production M5 scoring:
    M5 momentum + close/EMA20 = 10 setup points.

EMA_SCORE_5
    Remove the existing M5 contribution.
    Replace it with 5 points when M5 EMA20/EMA50 agrees.

EMA_SCORE_10
    Remove the existing M5 contribution.
    Replace it with 10 points when M5 EMA20/EMA50 agrees.

The experiment does NOT make M5 mandatory.

Instead, the score threshold determines which
otherwise-executable trades are accepted.

Thresholds:
    65 = NORMAL or better
    75 = STRONG or better
    85 = PREMIUM

No production files were modified.
No Git commits were created.
""")
