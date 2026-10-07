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
# Build exact executable decision scores
# ---------------------------------------------------------------------

score_arr = np.full(
    len(h1),
    np.nan,
)

long_arr = np.zeros(
    len(h1),
    dtype=bool,
)

short_arr = np.zeros(
    len(h1),
    dtype=bool,
)


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

    if decision.direction not in {
        "LONG",
        "SHORT",
    }:
        continue

    score_arr[i] = decision.score

    if decision.direction == "LONG":
        long_arr[i] = True
    else:
        short_arr[i] = True


# ---------------------------------------------------------------------
# Score threshold signal arrays
# ---------------------------------------------------------------------

thresholds = {
    "CURRENT": 0,
    "SCORE_75": 75,
    "SCORE_85": 85,
}


def threshold_signals(threshold):

    accepted = (
        np.isfinite(score_arr)
        & (score_arr >= threshold)
    )

    return (
        long_arr & accepted,
        short_arr & accepted,
    )


# ---------------------------------------------------------------------
# Backtest helper
# ---------------------------------------------------------------------

def run_backtest(
    name,
    threshold,
    start_time,
    end_time,
):

    long_signals, short_signals = (
        threshold_signals(threshold)
    )

    window_mask = (
        (decision_times >= start_time)
        & (decision_times < end_time)
    )

    long_period = (
        long_signals
        & window_mask
    )

    short_period = (
        short_signals
        & window_mask
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
        "win": (
            (p > 0).mean()
            * 100
        ),
        "pf": pf,
        "dd": float(dd.max()),
    }


# ---------------------------------------------------------------------
# Validation periods
# ---------------------------------------------------------------------

periods = [
    (
        "2022",
        pd.Timestamp(
            "2022-07-05",
            tz="UTC",
        ),
        pd.Timestamp(
            "2023-01-01",
            tz="UTC",
        ),
    ),
    (
        "2023",
        pd.Timestamp(
            "2023-01-01",
            tz="UTC",
        ),
        pd.Timestamp(
            "2024-01-01",
            tz="UTC",
        ),
    ),
    (
        "2024",
        pd.Timestamp(
            "2024-01-01",
            tz="UTC",
        ),
        pd.Timestamp(
            "2025-01-01",
            tz="UTC",
        ),
    ),
    (
        "2025",
        pd.Timestamp(
            "2025-01-01",
            tz="UTC",
        ),
        pd.Timestamp(
            "2026-01-01",
            tz="UTC",
        ),
    ),
    (
        "2026_YTD",
        pd.Timestamp(
            "2026-01-01",
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

all_results = []

for period_name, start_time, end_time in periods:

    for variant, threshold in thresholds.items():

        r = run_backtest(
            variant,
            threshold,
            start_time,
            end_time,
        )

        all_results.append(
            (
                period_name,
                r,
            )
        )


# ---------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("DECISION SCORE THRESHOLD OOS TEST")
print("=" * 125)

print()
print(
    f"{'PERIOD':13s} "
    f"{'VARIANT':12s} "
    f"{'SIGNALS':>8s} "
    f"{'TRADES':>7s} "
    f"{'NET':>10s} "
    f"{'RETURN':>9s} "
    f"{'WIN':>8s} "
    f"{'PF':>7s} "
    f"{'DD':>10s}"
)

print("-" * 125)

for period_name, r in all_results:

    pf = r["pf"]

    if np.isnan(pf):
        pf_text = "N/A"
    elif np.isinf(pf):
        pf_text = "INF"
    else:
        pf_text = f"{pf:.3f}"

    print(
        f"{period_name:13s} "
        f"{r['variant']:12s} "
        f"{r['signals']:8d} "
        f"{r['trades']:7d} "
        f"${r['net']:9.2f} "
        f"{r['return']:8.2f}% "
        f"{r['win']:7.2f}% "
        f"{pf_text:>7s} "
        f"${r['dd']:9.2f}"
    )

print()
print("=" * 125)
print("THRESHOLD DELTAS VS CURRENT")
print("=" * 125)

result_map = {
    (
        period,
        r["variant"],
    ): r
    for period, r in all_results
}

for period, _, _ in periods:

    base = result_map[
        (period, "CURRENT")
    ]

    print()
    print(period)

    for variant in (
        "SCORE_75",
        "SCORE_85",
    ):

        r = result_map[
            (period, variant)
        ]

        delta = (
            r["return"]
            - base["return"]
        )

        if (
            np.isfinite(r["pf"])
            and np.isfinite(base["pf"])
        ):
            pf_delta = (
                r["pf"]
                - base["pf"]
            )
            pf_text = (
                f"{pf_delta:+.3f}"
            )
        else:
            pf_text = "N/A"

        print(
            f"  {variant:10s} "
            f"return_delta={delta:+.2f}%  "
            f"PF_delta={pf_text}"
        )


# ---------------------------------------------------------------------
# Score 85 summary for 2025/2026
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("PREMIUM (85+) SUMMARY")
print("=" * 125)

for period in (
    "2025",
    "2026_YTD",
):

    current = result_map[
        (period, "CURRENT")
    ]

    premium = result_map[
        (period, "SCORE_85")
    ]

    print()
    print(period)

    print(
        f"  CURRENT : "
        f"trades={current['trades']}  "
        f"return={current['return']:+.2f}%  "
        f"PF={current['pf']:.3f}  "
        f"DD=${current['dd']:.2f}"
    )

    print(
        f"  85+     : "
        f"trades={premium['trades']}  "
        f"return={premium['return']:+.2f}%  "
        f"PF={premium['pf']:.3f}  "
        f"DD=${premium['dd']:.2f}"
    )


print()
print("=" * 125)
print("IMPORTANT")
print("=" * 125)

print("""
CURRENT
    Existing executable strategy.

SCORE_75
    Only execute decisions with score >= 75.

SCORE_85
    Only execute decisions with score >= 85.

These thresholds are pre-declared and are not optimized
against individual validation periods.

The purpose is to determine whether the PREMIUM tier's
historical advantage survives when used as an actual
trade-selection rule.

No production files were modified.
No Git commits were created.
""")
