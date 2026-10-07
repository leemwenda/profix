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
from profx.mtf_decision import build_decision_snapshot, validate_snapshot
from profx.engine import MarketData, BacktestEngine


ROOT = Path(".")
DATA_DIR = ROOT / "python/data/mtf"
CFG = load_config(ROOT / "config/xauusd.toml")
SYMBOL = "XAUUSD"

mtf = load_mtf(DATA_DIR, SYMBOL)
h1 = mtf.frame("H1")

decision_times = h1.index + pd.Timedelta(hours=1)

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


variants = {
    "BASELINE": np.zeros(len(h1), dtype=bool),
    "M5_REQUIRED": np.zeros(len(h1), dtype=bool),
    "M5_M1_REQUIRED": np.zeros(len(h1), dtype=bool),
}

short_variants = {
    name: np.zeros(len(h1), dtype=bool)
    for name in variants
}

counts = {
    "current": 0,
    "m5_valid": 0,
    "m1_valid": 0,
    "m5_and_m1_valid": 0,
}

combos = {}


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

    snapshot = build_decision_snapshot(
        mtf,
        decision_time,
    )
    validate_snapshot(snapshot)

    h1_row = aligned["H1"].iloc[i]

    spread = h1_row.get("spread")
    spread = None if pd.isna(spread) else float(spread)

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

    counts["current"] += 1

    m5 = bool(setup.m5_valid)
    m1 = bool(setup.m1_valid)

    if m5:
        counts["m5_valid"] += 1

    if m1:
        counts["m1_valid"] += 1

    if m5 and m1:
        counts["m5_and_m1_valid"] += 1

    key = (bool(setup.m15_valid), m5, m1)
    combos[key] = combos.get(key, 0) + 1

    direction = decision.direction

    # Current strategy
    if direction == "LONG":
        variants["BASELINE"][i] = True
    elif direction == "SHORT":
        short_variants["BASELINE"][i] = True

    # M5 required
    if m5:
        if direction == "LONG":
            variants["M5_REQUIRED"][i] = True
        elif direction == "SHORT":
            short_variants["M5_REQUIRED"][i] = True

    # M5 + M1 required
    if m5 and m1:
        if direction == "LONG":
            variants["M5_M1_REQUIRED"][i] = True
        elif direction == "SHORT":
            short_variants["M5_M1_REQUIRED"][i] = True


print()
print("=" * 80)
print("ACTUAL MTF CONFIRMATION ABLATION")
print("=" * 80)

print(f"Current executable signals : {counts['current']:,}")
print(f"M5 confirmation valid      : {counts['m5_valid']:,}")
print(f"M1 trigger valid           : {counts['m1_valid']:,}")
print(f"M5 + M1 valid              : {counts['m5_and_m1_valid']:,}")

print()
print("Validity combinations:")
for key, value in sorted(combos.items(), key=lambda x: -x[1]):
    print(f"  M15={key[0]} M5={key[1]} M1={key[2]} : {value:,}")


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
        "return": (final_equity / result.initial_balance - 1) * 100,
        "win_rate": float((p > 0).mean() * 100),
        "pf": gross_win / gross_loss if gross_loss else float("inf"),
        "max_dd": max_dd,
        "halted": result.halted,
    }


results = []

for name in variants:
    results.append(
        run_backtest(
            name,
            variants[name],
            short_variants[name],
        )
    )


print()
print("=" * 80)
print("BACKTEST RESULTS")
print("=" * 80)

for r in results:
    print(
        f"{r['variant']:20s} "
        f"signals={r['signals']:5d}  "
        f"trades={r['trades']:4d}  "
        f"net=${r['net']:8.2f}  "
        f"return={r['return']:7.2f}%  "
        f"win={r['win_rate']:6.2f}%  "
        f"PF={r['pf']:6.3f}  "
        f"DD=${r['max_dd']:7.2f}  "
        f"halted={r['halted']}"
    )

print()
print("=" * 80)
print("NO PRODUCTION FILES WERE MODIFIED")
print("=" * 80)
