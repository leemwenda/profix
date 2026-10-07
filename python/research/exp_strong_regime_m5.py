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

baseline_long = np.zeros(len(h1), dtype=bool)
baseline_short = np.zeros(len(h1), dtype=bool)

strong_long = np.zeros(len(h1), dtype=bool)
strong_short = np.zeros(len(h1), dtype=bool)

strong_m5_long = np.zeros(len(h1), dtype=bool)
strong_m5_short = np.zeros(len(h1), dtype=bool)


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

regime_score_arr = np.full(
    len(h1),
    np.nan,
)

m5_confirmation_arr = np.zeros(
    len(h1),
    dtype=bool,
)


# ---------------------------------------------------------------------
# Strong regime threshold
#
# assess_decision() normalizes regime strength to 0..35.
# We define >=25/35 as a strong regime.
# This is declared BEFORE looking at the quarter results.
# ---------------------------------------------------------------------

STRONG_REGIME_MIN = 25


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
    # Baseline
    # -------------------------------------------------------------

    if direction == "LONG":
        baseline_long[i] = True
    else:
        baseline_short[i] = True

    # -------------------------------------------------------------
    # Strong-regime condition
    # -------------------------------------------------------------

    regime_score = int(
        decision.regime_score
    )

    regime_score_arr[i] = regime_score

    strong_regime = (
        regime_score
        >= STRONG_REGIME_MIN
    )

    # -------------------------------------------------------------
    # Current production M5 confirmation
    # -------------------------------------------------------------

    m5_confirmed = bool(
        setup.m5_valid
    )

    m5_confirmation_arr[i] = (
        m5_confirmed
    )

    # -------------------------------------------------------------
    # Strong regime only
    # -------------------------------------------------------------

    if strong_regime:

        if direction == "LONG":
            strong_long[i] = True
        else:
            strong_short[i] = True

    # -------------------------------------------------------------
    # Strong regime + M5 confirmation
    # -------------------------------------------------------------

    if (
        strong_regime
        and m5_confirmed
    ):

        if direction == "LONG":
            strong_m5_long[i] = True
        else:
            strong_m5_short[i] = True


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
# Run
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

    strong = run_backtest(
        "STRONG_REGIME",
        strong_long,
        strong_short,
        start_time,
        end_time,
    )

    strong_m5 = run_backtest(
        "STRONG_REGIME_M5",
        strong_m5_long,
        strong_m5_short,
        start_time,
        end_time,
    )

    all_results.append(
        (
            period_name,
            baseline,
            strong,
            strong_m5,
        )
    )


# ---------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("STRONG REGIME + M5 CONFIRMATION OOS EXPERIMENT")
print("=" * 120)

print()
print(
    f"Strong regime threshold : "
    f"{STRONG_REGIME_MIN}/35"
)

print(
    f"Executable signals      : "
    f"{int(np.isfinite(regime_score_arr).sum())}"
)

print(
    f"Strong regime signals   : "
    f"{int(strong_long.sum() + strong_short.sum())}"
)

print(
    f"Strong + M5 signals     : "
    f"{int(strong_m5_long.sum() + strong_m5_short.sum())}"
)

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

print("-" * 120)

for period_name, baseline, strong, strong_m5 in all_results:

    for r in (
        baseline,
        strong,
        strong_m5,
    ):

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

    print("-" * 120)


# ---------------------------------------------------------------------
# Deltas vs baseline
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("DELTAS VS BASELINE")
print("=" * 120)

for period_name, baseline, strong, strong_m5 in all_results:

    print()
    print(period_name)

    for candidate in (
        strong,
        strong_m5,
    ):

        return_delta = (
            candidate["return"]
            - baseline["return"]
        )

        if (
            np.isfinite(candidate["pf"])
            and np.isfinite(baseline["pf"])
        ):
            pf_delta = (
                candidate["pf"]
                - baseline["pf"]
            )
            pf_text = f"{pf_delta:+.3f}"
        else:
            pf_text = "N/A"

        print(
            f"  {candidate['variant']:20s} "
            f"return_delta={return_delta:+.2f}%  "
            f"PF_delta={pf_text}  "
            f"trades="
            f"{baseline['trades']} -> "
            f"{candidate['trades']}"
        )


# ---------------------------------------------------------------------
# M5 incremental value INSIDE strong regime
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("M5 INCREMENTAL VALUE INSIDE STRONG REGIME")
print("=" * 120)

for period_name, baseline, strong, strong_m5 in all_results:

    print()
    print(period_name)

    delta = (
        strong_m5["return"]
        - strong["return"]
    )

    if (
        np.isfinite(strong_m5["pf"])
        and np.isfinite(strong["pf"])
    ):
        pf_delta = (
            strong_m5["pf"]
            - strong["pf"]
        )
        pf_text = f"{pf_delta:+.3f}"
    else:
        pf_text = "N/A"

    print(
        f"  Strong regime      : "
        f"return={strong['return']:+.2f}%  "
        f"PF={strong['pf']:.3f}  "
        f"trades={strong['trades']}"
    )

    print(
        f"  Strong + M5        : "
        f"return={strong_m5['return']:+.2f}%  "
        f"PF={strong_m5['pf']:.3f}  "
        f"trades={strong_m5['trades']}"
    )

    print(
        f"  M5 incremental     : "
        f"return_delta={delta:+.2f}%  "
        f"PF_delta={pf_text}"
    )


# ---------------------------------------------------------------------
# Aggregate OOS comparison
# ---------------------------------------------------------------------

print()
print("=" * 120)
print("AGGREGATE OOS COMPARISON")
print("=" * 120)

for key, label in (
    (1, "STRONG_REGIME"),
    (2, "STRONG_REGIME_M5"),
):

    compounded = 1.0
    total_net = 0.0
    positive = 0
    negative = 0

    for item in all_results:

        r = item[key]

        compounded *= (
            1
            + r["return"]
            / 100
        )

        total_net += r["net"]

        if r["return"] > 0:
            positive += 1
        elif r["return"] < 0:
            negative += 1

    compounded_return = (
        compounded - 1
    ) * 100

    print(
        f"{label:22s} "
        f"compounded="
        f"{compounded_return:+.2f}%  "
        f"sum_net="
        f"${total_net:+.2f}  "
        f"positive={positive}  "
        f"negative={negative}"
    )


print()
print("=" * 120)
print("M5 ALIGNMENT RATE")
print("=" * 120)

valid = np.isfinite(
    regime_score_arr
)

strong_mask = (
    valid
    & (
        regime_score_arr
        >= STRONG_REGIME_MIN
    )
)

print(
    f"Strong regime decisions : "
    f"{int(strong_mask.sum())}"
)

if strong_mask.sum():

    print(
        f"M5 confirmed within strong regime : "
        f"{m5_confirmation_arr[strong_mask].mean() * 100:.2f}%"
    )


print()
print("=" * 120)
print("INTERPRETATION")
print("=" * 120)

print("""
BASELINE
    Existing executable strategy.

STRONG_REGIME
    Existing strategy, but only when the normalized
    regime score is at least 25/35.

STRONG_REGIME_M5
    Existing strategy, restricted to strong regimes,
    AND requiring the current production M5 confirmation.

The strong-regime threshold is declared in advance.

The purpose is to determine whether M5 becomes useful
as an entry gate specifically when the higher-timeframe
regime is already strong.

No production files were modified.
No Git commits were created.
""")
