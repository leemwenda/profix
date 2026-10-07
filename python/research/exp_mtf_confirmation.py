from pathlib import Path
import numpy as np

from profx.config import load_config
from profx.mtf import load_mtf
from profx.mtf_backtest import build_mtf_signals
from profx.engine import BacktestEngine
from profx.data import load_market_data


ROOT = Path(".")
CFG = load_config(ROOT / "config/xauusd.toml")

mtf = load_mtf(
    ROOT / "python/data/mtf",
    "XAUUSD",
)

signals, records = build_mtf_signals(
    mtf=mtf,
    cfg=CFG,
    symbol="XAUUSD",
)

print("=" * 90)
print("MTF CONFIRMATION ABLATION")
print("=" * 90)

# ---------------------------------------------------------------------
# Inspect setup-score distribution
# ---------------------------------------------------------------------

setup_scores = {}

for r in records:
    score = r.decision.setup_score
    setup_scores[score] = setup_scores.get(score, 0) + 1

print()
print("SETUP SCORE DISTRIBUTION")
for score in sorted(setup_scores):
    print(f"setup_score={score:>2}: {setup_scores[score]:>6} records")

print()
print("INTERPRETATION")
print("15 = M15 only")
print("25 = M15 + M5")
print("30 = M15 + M5 + M1")

# ---------------------------------------------------------------------
# Build signal variants
# ---------------------------------------------------------------------

def filter_signals(min_setup_score):
    filtered = {}

    for symbol, (sig, ts) in signals.items():
        sig2 = sig.copy()

        # Map each signal timestamp to its decision record.
        record_by_time = {
            r.signal_time: r
            for r in records
        }

        for i in range(len(sig2)):
            if sig2[i] == 0:
                continue

            record = record_by_time.get(ts[i])

            if record is None:
                sig2[i] = 0
                continue

            if record.decision.setup_score < min_setup_score:
                sig2[i] = 0

        filtered[symbol] = (sig2, ts)

    return filtered


# ---------------------------------------------------------------------
# Load existing H1 market data exactly as the normal backtest does
# ---------------------------------------------------------------------

market = load_market_data(
    ROOT / "python/data",
    ["XAUUSD"],
)

# ---------------------------------------------------------------------
# Run one experiment
# ---------------------------------------------------------------------

def run_variant(name, min_setup_score):
    variant_signals = filter_signals(min_setup_score)

    engine = BacktestEngine(
        market,
        CFG,
        signal_override=variant_signals,
    )

    result = engine.run()

    print()
    print("-" * 90)
    print(name)
    print("-" * 90)

    print(f"Minimum setup score : {min_setup_score}")
    print(f"Final equity        : ${result.final_equity:,.2f}")
    print(f"Net P/L             : ${result.net_pnl:,.2f}")
    print(f"Return              : {result.return_pct:.2f}%")
    print(f"Trades              : {len(result.trades)}")
    print(f"Win rate            : {result.win_rate * 100:.2f}%")
    print(f"Profit factor       : {result.profit_factor:.3f}")
    print(f"Max drawdown        : {result.max_drawdown_pct:.2f}%")
    print(f"Halted              : {result.halted}")
    print(f"Halt reason         : {result.halt_reason}")

    return result


# ---------------------------------------------------------------------
# Run all three variants
# ---------------------------------------------------------------------

results = {}

results["CURRENT_M15"] = run_variant(
    "CURRENT — M15 mandatory, M5/M1 optional",
    15,
)

results["REQUIRE_M5"] = run_variant(
    "REQUIRE M5 — M15 + M5 confirmation",
    25,
)

results["REQUIRE_M5_M1"] = run_variant(
    "REQUIRE M5 + M1 — full lower-timeframe confirmation",
    30,
)

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)

print(
    f"{'Variant':<25}"
    f"{'Trades':>8}"
    f"{'Return':>12}"
    f"{'PF':>10}"
    f"{'Win%':>10}"
    f"{'Max DD':>10}"
)

for name, result in results.items():
    print(
        f"{name:<25}"
        f"{len(result.trades):>8}"
        f"{result.return_pct:>11.2f}%"
        f"{result.profit_factor:>10.3f}"
        f"{result.win_rate * 100:>9.2f}%"
        f"{result.max_drawdown_pct:>9.2f}%"
    )

print()
print("=" * 90)
print("NO PRODUCTION CODE WAS CHANGED")
print("=" * 90)
