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
# Align closed candles
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
# Build baseline signals and custom score models
# ---------------------------------------------------------------------

long_arr = np.zeros(len(h1), dtype=bool)
short_arr = np.zeros(len(h1), dtype=bool)

scores = {
    "CURRENT": np.full(len(h1), np.nan),
    "NO_M1": np.full(len(h1), np.nan),
    "NO_M1_RENORM": np.full(len(h1), np.nan),
}


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

    if not (
        decision.executable
        and market_ok
    ):
        continue

    direction = decision.direction

    if direction == "LONG":
        long_arr[i] = True
    elif direction == "SHORT":
        short_arr[i] = True
    else:
        continue

    # -------------------------------------------------------------
    # Existing points
    # -------------------------------------------------------------

    m15_ok = setup.m15_valid
    m5_ok = setup.m5_valid
    m1_ok = setup.m1_valid

    regime_points = decision.regime_score

    market_points = (
        decision.volatility_score
        + decision.spread_score
        + decision.session_score
    )

    m15_points = 15 if m15_ok else 0
    m5_points = 10 if m5_ok else 0
    m1_points = 5 if m1_ok else 0

    # -------------------------------------------------------------
    # A: CURRENT
    #
    # setup = M15 15 + M5 10 + M1 5
    # normalized to 50
    # -------------------------------------------------------------

    current_setup = (
        m15_points
        + m5_points
        + m1_points
    )

    current_setup_norm = round(
        (current_setup / 30.0) * 50
    )

    scores["CURRENT"][i] = max(
        0,
        min(
            100,
            regime_points
            + current_setup_norm
            + market_points,
        ),
    )

    # -------------------------------------------------------------
    # B: NO M1, NO RENORMALIZATION
    #
    # M15 15 + M5 10
    # M1 = 0
    #
    # Still uses the original 30-point setup scale.
    # -------------------------------------------------------------

    no_m1_setup = (
        m15_points
        + m5_points
    )

    no_m1_norm = round(
        (no_m1_setup / 30.0) * 50
    )

    scores["NO_M1"][i] = max(
        0,
        min(
            100,
            regime_points
            + no_m1_norm
            + market_points,
        ),
    )

    # -------------------------------------------------------------
    # C: NO M1 + RENORMALIZE
    #
    # Remaining setup:
    # M15 15 + M5 10 = 25 max
    #
    # Reweight remaining setup to full 50:
    # M15 becomes 30
    # M5 becomes 20
    # -------------------------------------------------------------

    renorm_setup = (
        m15_points
        + m5_points
    )

    renorm_norm = round(
        (renorm_setup / 25.0) * 50
    )

    scores["NO_M1_RENORM"][i] = max(
        0,
        min(
            100,
            regime_points
            + renorm_norm
            + market_points,
        ),
    )


# ---------------------------------------------------------------------
# Backtest helper
# ---------------------------------------------------------------------

def run_backtest(
    score_array,
    threshold,
    start_time,
    end_time,
    name,
):

    period_mask = (
        (decision_times >= start_time)
        & (decision_times < end_time)
    )

    accepted = (
        np.isfinite(score_array)
        & (score_array >= threshold)
    )

    long_period = (
        long_arr
        & accepted
        & period_mask
    )

    short_period = (
        short_arr
        & accepted
        & period_mask
    )

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
        window=(
            start_time,
            end_time,
        ),
    ).run()

    trades = pd.DataFrame(
        result.trades
    )

    signals = int(
        long_period.sum()
        + short_period.sum()
    )

    final_equity = float(
        result.equity.iloc[-1]
    )

    pnl = (
        final_equity
        - result.initial_balance
    )

    if trades.empty:

        return {
            "variant": name,
            "signals": signals,
            "trades": 0,
            "net": pnl,
            "return": 0.0,
            "win": 0.0,
            "pf": float("nan"),
            "dd": 0.0,
        }

    p = pd.to_numeric(
        trades["pnl"],
        errors="coerce",
    ).dropna()

    wins = p[p > 0]
    losses = p[p < 0]

    gross_win = float(wins.sum())
    gross_loss = abs(float(losses.sum()))

    pf = (
        gross_win / gross_loss
        if gross_loss
        else float("inf")
    )

    equity = pd.Series(
        result.equity
    ).astype(float)

    dd = (
        equity.cummax()
        - equity
    )

    return {
        "variant": name,
        "signals": signals,
        "trades": len(p),
        "net": pnl,
        "return": (
            final_equity
            / result.initial_balance
            - 1
        ) * 100,
        "win": float(
            (p > 0).mean() * 100
        ),
        "pf": pf,
        "dd": float(dd.max()),
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
# Run
# ---------------------------------------------------------------------

results = []

for period_name, start_time, end_time in periods:

    for model_name, score_array in scores.items():

        for threshold in (75, 85):

            r = run_backtest(
                score_array,
                threshold,
                start_time,
                end_time,
                f"{model_name}_{threshold}",
            )

            results.append(
                (
                    period_name,
                    r,
                )
            )


# ---------------------------------------------------------------------
# Full table
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("FINAL M1 ABLATION: REMOVE VS RENORMALIZE")
print("=" * 125)

print()
print(
    f"{'PERIOD':13s} "
    f"{'VARIANT':20s} "
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
        f"{r['variant']:20s} "
        f"{r['signals']:8d} "
        f"{r['trades']:7d} "
        f"${r['net']:9.2f} "
        f"{r['return']:8.2f}% "
        f"{r['win']:7.2f}% "
        f"{pf_text:>7s} "
        f"${r['dd']:9.2f}"
    )


# ---------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------

result_map = {
    (
        period,
        r["variant"],
    ): r
    for period, r in results
}


print()
print("=" * 125)
print("AGGREGATE COMPARISON")
print("=" * 125)

for threshold in (75, 85):

    print()
    print(
        f"THRESHOLD {threshold}"
    )

    for model in (
        "CURRENT",
        "NO_M1",
        "NO_M1_RENORM",
    ):

        values = [
            result_map[
                (
                    period,
                    f"{model}_{threshold}",
                )
            ]
            for period, _, _ in periods
        ]

        compounded = 1.0
        net = 0.0
        positive = 0
        negative = 0

        for r in values:

            compounded *= (
                1
                + r["return"]
                / 100
            )

            net += r["net"]

            if r["return"] > 0:
                positive += 1
            elif r["return"] < 0:
                negative += 1

        compounded_return = (
            compounded - 1
        ) * 100

        print(
            f"  {model:16s} "
            f"compounded="
            f"{compounded_return:+.2f}%  "
            f"sum_net="
            f"${net:+.2f}  "
            f"positive={positive}  "
            f"negative={negative}"
        )


# ---------------------------------------------------------------------
# Score distribution
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("SCORE DISTRIBUTION COMPARISON")
print("=" * 125)

for model, score_array in scores.items():

    valid = np.isfinite(score_array)

    values = score_array[valid]

    print()
    print(
        f"{model:16s} "
        f"mean={values.mean():.2f}  "
        f"median={np.median(values):.2f}  "
        f"75th={np.quantile(values, .75):.2f}  "
        f"90th={np.quantile(values, .90):.2f}"
    )

    for threshold in (75, 85):

        print(
            f"  >= {threshold}: "
            f"{int((values >= threshold).sum())}"
        )


print()
print("=" * 125)
print("DECISION")
print("=" * 125)

print("""
CURRENT:
    M15=15, M5=10, M1=5
    normalized to 50 setup points.

NO_M1:
    M15=15, M5=10, M1=0
    original 30-point setup scale retained.

NO_M1_RENORM:
    M15=30, M5=20, M1=0
    setup renormalized to the full 50 points.

The test separates:
    removing M1
from:
    reallocating M1's score weight.

No production files were modified.
No Git commits were created.
""")
