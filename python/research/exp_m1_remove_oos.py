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

for timeframe in (
    "H1",
    "M15",
    "M5",
    "M1",
):

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
# Build exact executable baseline signals
# and calculate two score models.
# ---------------------------------------------------------------------

long_arr = np.zeros(
    len(h1),
    dtype=bool,
)

short_arr = np.zeros(
    len(h1),
    dtype=bool,
)

current_score = np.full(
    len(h1),
    np.nan,
)

no_m1_score = np.full(
    len(h1),
    np.nan,
)


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

    # -------------------------------------------------------------
    # Current score
    # -------------------------------------------------------------

    current_score[i] = decision.score

    # -------------------------------------------------------------
    # Remove M1 from the setup and renormalize the remaining
    # M15 + M5 setup from 25 points back to 50.
    #
    # M15 = 15
    # M5  = 10
    # M1  =  5
    #
    # Without M1:
    # M15 + M5 = 25 max
    # Renormalized to 50.
    # -------------------------------------------------------------

    setup_without_m1 = (
        setup.score
        - (
            5
            if setup.m1_valid
            else 0
        )
    )

    setup_without_m1 = max(
        0,
        setup_without_m1,
    )

    no_m1_setup_points = round(
        (
            setup_without_m1
            / 25.0
        ) * 50
    )

    no_m1_score[i] = max(
        0,
        min(
            100,
            int(
                decision.regime_score
                + no_m1_setup_points
                + decision.volatility_score
                + decision.spread_score
                + decision.session_score
            ),
        ),
    )


# ---------------------------------------------------------------------
# Backtest helper
# ---------------------------------------------------------------------

def run_backtest(
    name,
    score_array,
    threshold,
    start_time,
    end_time,
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

    equity = pd.Series(
        result.equity
    ).astype(float)

    drawdown = (
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
        "dd": float(
            drawdown.max()
        ),
    }


# ---------------------------------------------------------------------
# Validation periods
#
# M5/M1-based analysis is meaningful from 2025 onward.
# ---------------------------------------------------------------------

periods = [
    (
        "2025_Q2",
        pd.Timestamp(
            "2025-04-30",
            tz="UTC",
        ),
        pd.Timestamp(
            "2025-07-01",
            tz="UTC",
        ),
    ),
    (
        "2025_Q3",
        pd.Timestamp(
            "2025-07-01",
            tz="UTC",
        ),
        pd.Timestamp(
            "2025-10-01",
            tz="UTC",
        ),
    ),
    (
        "2025_Q4",
        pd.Timestamp(
            "2025-10-01",
            tz="UTC",
        ),
        pd.Timestamp(
            "2026-01-01",
            tz="UTC",
        ),
    ),
    (
        "2026_Q1",
        pd.Timestamp(
            "2026-01-01",
            tz="UTC",
        ),
        pd.Timestamp(
            "2026-04-01",
            tz="UTC",
        ),
    ),
    (
        "2026_Q2",
        pd.Timestamp(
            "2026-04-01",
            tz="UTC",
        ),
        pd.Timestamp(
            "2026-07-01",
            tz="UTC",
        ),
    ),
    (
        "2026_Q3",
        pd.Timestamp(
            "2026-07-01",
            tz="UTC",
        ),
        pd.Timestamp(
            "2026-10-01",
            tz="UTC",
        ),
    ),
    (
        "2026_Q4_YTD",
        pd.Timestamp(
            "2026-10-01",
            tz="UTC",
        ),
        pd.Timestamp(
            "2027-01-01",
            tz="UTC",
        ),
    ),
]


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

results = []

for period_name, start_time, end_time in periods:

    # Current score model
    for threshold in (
        75,
        85,
    ):

        results.append(
            (
                period_name,
                run_backtest(
                    f"CURRENT_{threshold}",
                    current_score,
                    threshold,
                    start_time,
                    end_time,
                ),
            )
        )

        # No-M1 score model
        results.append(
            (
                period_name,
                run_backtest(
                    f"NO_M1_{threshold}",
                    no_m1_score,
                    threshold,
                    start_time,
                    end_time,
                ),
            )
        )


# ---------------------------------------------------------------------
# Print full results
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("M1 REMOVAL + SCORE RENORMALIZATION OOS TEST")
print("=" * 120)

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

print("-" * 120)

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
        f"{r['variant']:15s} "
        f"{r['signals']:8d} "
        f"{r['trades']:7d} "
        f"${r['net']:9.2f} "
        f"{r['return']:8.2f}% "
        f"{r['win']:7.2f}% "
        f"{pf_text:>7s} "
        f"${r['dd']:9.2f}"
    )


# ---------------------------------------------------------------------
# Deltas
# ---------------------------------------------------------------------

result_map = {
    (
        period,
        r["variant"],
    ): r
    for period, r in results
}


print()
print("=" * 120)
print("NO-M1 DELTA VS CURRENT")
print("=" * 120)

for period_name, _, _ in periods:

    print()
    print(period_name)

    for threshold in (
        75,
        85,
    ):

        current = result_map[
            (
                period_name,
                f"CURRENT_{threshold}",
            )
        ]

        no_m1 = result_map[
            (
                period_name,
                f"NO_M1_{threshold}",
            )
        ]

        return_delta = (
            no_m1["return"]
            - current["return"]
        )

        pf_delta = float("nan")

        if (
            np.isfinite(
                no_m1["pf"]
            )
            and np.isfinite(
                current["pf"]
            )
        ):
            pf_delta = (
                no_m1["pf"]
                - current["pf"]
            )

        if np.isnan(pf_delta):
            pf_text = "N/A"
        else:
            pf_text = f"{pf_delta:+.3f}"

        print(
            f"  THRESHOLD {threshold}: "
            f"return_delta={return_delta:+.2f}%  "
            f"PF_delta={pf_text}  "
            f"trades "
            f"{current['trades']} -> "
            f"{no_m1['trades']}"
        )


# ---------------------------------------------------------------------
# Score movement
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("SCORE MOVEMENT FROM REMOVING M1")
print("=" * 120)

valid = (
    np.isfinite(current_score)
    & np.isfinite(no_m1_score)
)

delta = (
    no_m1_score[valid]
    - current_score[valid]
)

print(
    f"Signals with scores : {int(valid.sum())}"
)

print(
    f"Average score change: "
    f"{float(delta.mean()):+.2f}"
)

print(
    f"Minimum change      : "
    f"{float(delta.min()):+.0f}"
)

print(
    f"Maximum change      : "
    f"{float(delta.max()):+.0f}"
)

print(
    f"Signals moved up    : "
    f"{int((delta > 0).sum())}"
)

print(
    f"Signals unchanged   : "
    f"{int((delta == 0).sum())}"
)

print(
    f"Signals moved down  : "
    f"{int((delta < 0).sum())}"
)


# ---------------------------------------------------------------------
# Aggregate comparison
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("AGGREGATE OOS COMPARISON")
print("=" * 120)

for threshold in (
    75,
    85,
):

    print()
    print(
        f"THRESHOLD {threshold}"
    )

    for label in (
        "CURRENT",
        "NO_M1",
    ):

        subset = [
            r
            for period, r in results
            if r["variant"]
            == f"{label}_{threshold}"
        ]

        compounded = 1.0
        total_net = 0.0
        positive_periods = 0
        negative_periods = 0

        for r in subset:

            compounded *= (
                1
                + r["return"]
                / 100
            )

            total_net += r["net"]

            if r["return"] > 0:
                positive_periods += 1
            elif r["return"] < 0:
                negative_periods += 1

        compounded_return = (
            compounded - 1
        ) * 100

        print(
            f"  {label:8s} "
            f"compounded="
            f"{compounded_return:+.2f}%  "
            f"sum_net="
            f"${total_net:+.2f}  "
            f"positive="
            f"{positive_periods}  "
            f"negative="
            f"{negative_periods}"
        )


print()
print("=" * 120)
print("INTERPRETATION")
print("=" * 120)

print("""
CURRENT_75 / CURRENT_85
    Existing score including the current M1 contribution.

NO_M1_75 / NO_M1_85
    M1 removed from the setup.
    Remaining M15 + M5 setup points are renormalized
    from 25 back to the full 50-point setup allocation.

IMPORTANT:
    M1 removal only changes actual trade selection here because
    we explicitly apply a 75+ or 85+ score threshold.

This is therefore a research test of:
    score model + threshold
rather than a claim about the current un-gated production engine.

No production files were modified.
No Git commits were created.
""")
