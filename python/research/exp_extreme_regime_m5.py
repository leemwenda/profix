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
# Signal arrays
# ---------------------------------------------------------------------

baseline_long = np.zeros(
    len(h1),
    dtype=bool,
)

baseline_short = np.zeros(
    len(h1),
    dtype=bool,
)


# Extreme-regime + M5 arrays for each threshold

thresholds = (30, 32, 34)

extreme_long = {
    threshold: np.zeros(
        len(h1),
        dtype=bool,
    )
    for threshold in thresholds
}

extreme_short = {
    threshold: np.zeros(
        len(h1),
        dtype=bool,
    )
    for threshold in thresholds
}

extreme_m5_long = {
    threshold: np.zeros(
        len(h1),
        dtype=bool,
    )
    for threshold in thresholds
}

extreme_m5_short = {
    threshold: np.zeros(
        len(h1),
        dtype=bool,
    )
    for threshold in thresholds
}


regime_score_arr = np.full(
    len(h1),
    np.nan,
)

m5_confirmation_arr = np.zeros(
    len(h1),
    dtype=bool,
)


# ---------------------------------------------------------------------
# Reconstruct exact executable pipeline
# ---------------------------------------------------------------------

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

    regime = assess_regime(
        features
    )

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

    # -------------------------------------------------------------
    # BASELINE
    # -------------------------------------------------------------

    if direction == "LONG":
        baseline_long[i] = True
    else:
        baseline_short[i] = True

    # -------------------------------------------------------------
    # Normalized regime strength
    # -------------------------------------------------------------

    regime_score = int(
        decision.regime_score
    )

    regime_score_arr[i] = regime_score

    # -------------------------------------------------------------
    # Current production M5 confirmation
    # -------------------------------------------------------------

    m5_confirmed = bool(
        setup.m5_valid
    )

    m5_confirmation_arr[i] = m5_confirmed

    # -------------------------------------------------------------
    # Extreme regime candidates
    # -------------------------------------------------------------

    for threshold in thresholds:

        strong = (
            regime_score
            >= threshold
        )

        if not strong:
            continue

        if direction == "LONG":
            extreme_long[threshold][i] = True
        else:
            extreme_short[threshold][i] = True

        # M5 is an additional gate here.
        if m5_confirmed:

            if direction == "LONG":
                extreme_m5_long[
                    threshold
                ][i] = True
            else:
                extreme_m5_short[
                    threshold
                ][i] = True


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

    window_mask = (
        (decision_times >= start_time)
        & (decision_times < end_time)
    )

    long_period = (
        long_arr
        & window_mask
    )

    short_period = (
        short_arr
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
        "win": float(
            (p > 0).mean() * 100
        ),
        "pf": pf,
        "dd": float(
            dd.max()
        ),
    }


# ---------------------------------------------------------------------
# Validation periods
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
# Run all thresholds
# ---------------------------------------------------------------------

results = []

for period_name, start_time, end_time in periods:

    baseline = run_backtest(
        "BASELINE",
        baseline_long,
        baseline_short,
        start_time,
        end_time,
    )

    results.append(
        (
            period_name,
            baseline,
        )
    )

    for threshold in thresholds:

        extreme = run_backtest(
            f"REGIME_{threshold}",
            extreme_long[threshold],
            extreme_short[threshold],
            start_time,
            end_time,
        )

        extreme_m5 = run_backtest(
            f"REGIME_{threshold}_M5",
            extreme_m5_long[threshold],
            extreme_m5_short[threshold],
            start_time,
            end_time,
        )

        results.append(
            (
                period_name,
                extreme,
            )
        )

        results.append(
            (
                period_name,
                extreme_m5,
            )
        )


# ---------------------------------------------------------------------
# Print main results
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("EXTREME REGIME + M5 OOS EXPERIMENT")
print("=" * 125)

print()
print(
    f"{'PERIOD':13s} "
    f"{'VARIANT':18s} "
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
        f"{r['variant']:18s} "
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
print("=" * 125)
print("M5 INCREMENTAL DELTA INSIDE EACH EXTREME REGIME")
print("=" * 125)

for threshold in thresholds:

    print()
    print(
        f"REGIME >= {threshold}/35"
    )

    for period_name, _, _ in periods:

        regime = result_map[
            (
                period_name,
                f"REGIME_{threshold}",
            )
        ]

        regime_m5 = result_map[
            (
                period_name,
                f"REGIME_{threshold}_M5",
            )
        ]

        return_delta = (
            regime_m5["return"]
            - regime["return"]
        )

        if (
            np.isfinite(regime["pf"])
            and np.isfinite(regime_m5["pf"])
        ):
            pf_delta = (
                regime_m5["pf"]
                - regime["pf"]
            )
            pf_text = f"{pf_delta:+.3f}"
        else:
            pf_text = "N/A"

        print(
            f"{period_name:13s} "
            f"return_delta={return_delta:+.2f}%  "
            f"PF_delta={pf_text}  "
            f"trades="
            f"{regime['trades']} -> "
            f"{regime_m5['trades']}"
        )


# ---------------------------------------------------------------------
# Aggregate comparison
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("AGGREGATE RESULTS")
print("=" * 125)

variants_to_aggregate = (
    "BASELINE",
    "REGIME_30",
    "REGIME_30_M5",
    "REGIME_32",
    "REGIME_32_M5",
    "REGIME_34",
    "REGIME_34_M5",
)

for variant in variants_to_aggregate:

    subset = [
        r
        for period, r in results
        if r["variant"] == variant
    ]

    if not subset:
        continue

    compounded = 1.0
    total_net = 0.0
    positive = 0
    negative = 0
    trades_total = 0

    for r in subset:

        compounded *= (
            1
            + r["return"]
            / 100
        )

        total_net += r["net"]
        trades_total += r["trades"]

        if r["return"] > 0:
            positive += 1
        elif r["return"] < 0:
            negative += 1

    compounded_return = (
        compounded - 1
    ) * 100

    print(
        f"{variant:18s} "
        f"compounded="
        f"{compounded_return:+.2f}%  "
        f"sum_net="
        f"${total_net:+.2f}  "
        f"total_trades="
        f"{trades_total}  "
        f"positive={positive}  "
        f"negative={negative}"
    )


# ---------------------------------------------------------------------
# Regime population
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("EXTREME REGIME POPULATION")
print("=" * 125)

valid = np.isfinite(
    regime_score_arr
)

values = regime_score_arr[valid]

print(
    f"Executable decisions : "
    f"{int(valid.sum())}"
)

for threshold in thresholds:

    mask = (
        values
        >= threshold
    )

    print(
        f">= {threshold}/35 : "
        f"{int(mask.sum())} "
        f"({mask.mean() * 100:.2f}%)"
    )


# ---------------------------------------------------------------------
# M5 confirmation inside extreme regimes
# ---------------------------------------------------------------------

print()
print("=" * 125)
print("M5 CONFIRMATION RATE INSIDE EXTREME REGIMES")
print("=" * 125)

for threshold in thresholds:

    mask = (
        valid
        & (
            regime_score_arr
            >= threshold
        )
    )

    count = int(
        mask.sum()
    )

    if count == 0:

        print(
            f">= {threshold}/35 : "
            f"no signals"
        )

        continue

    confirmed = int(
        m5_confirmation_arr[mask].sum()
    )

    print(
        f">= {threshold}/35 : "
        f"{confirmed}/{count} "
        f"= "
        f"{confirmed / count * 100:.2f}%"
    )


print()
print("=" * 125)
print("INTERPRETATION")
print("=" * 125)

print("""
REGIME_30 / 32 / 34
    Keep only executable signals whose normalized
    regime strength is at least the stated threshold.

REGIME_30_M5 / 32_M5 / 34_M5
    Same extreme-regime filter, plus the CURRENT
    production M5 confirmation.

This is intentionally testing the existing M5 confirmation,
not the rejected EMA20/EMA50 hard filter.

No production files were modified.
No Git commits were created.
""")
