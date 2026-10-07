from pathlib import Path
import numpy as np
import pandas as pd

from profx.config import load_config
from profx.mtf import load_mtf
from profx.mtf_backtest import build_mtf_signals
from profx.engine import MarketData, BacktestEngine


ROOT = Path(".")
DATA_DIR = ROOT / "python/data/mtf"
CFG = load_config(ROOT / "config/xauusd.toml")

mtf = load_mtf(DATA_DIR, "XAUUSD")

signals, records = build_mtf_signals(
    mtf=mtf,
    cfg=CFG,
    symbol="XAUUSD",
)

h1 = mtf.frame("H1")

# ------------------------------------------------------------------
# The existing adapter already produces the executable signal arrays.
#
# We now independently determine whether M5 and M1 were actually
# available at each H1 decision time by checking the native coverage.
#
# Strict MTF rule:
#   H1 direction
#   + valid M15
#   + valid M5
#   + valid M1
# ------------------------------------------------------------------

m5 = mtf.frame("M5")
m1 = mtf.frame("M1")
m15 = mtf.frame("M15")

def available(frame, decision_time):
    if frame.empty:
        return False
    # A candle at timestamp t is closed at t + its timeframe duration.
    # We need a genuinely closed native candle at/before decision_time.
    return frame.index.searchsorted(decision_time, side="right") > 0


strict_long = signals["XAUUSD"][0].copy()
strict_short = signals["XAUUSD"][1].copy()

removed = 0
kept = 0

for i, h1_open in enumerate(h1.index):
    decision_time = h1_open + pd.Timedelta(hours=1)

    # Only apply strictness where the original adapter produced a signal.
    if not (strict_long[i] or strict_short[i]):
        continue

    has_m15 = available(m15, decision_time)
    has_m5 = available(m5, decision_time)
    has_m1 = available(m1, decision_time)

    # True strict-MTF availability requirement.
    if not (has_m15 and has_m5 and has_m1):
        strict_long[i] = False
        strict_short[i] = False
        removed += 1
    else:
        kept += 1

print("=" * 78)
print("PROFX STRICT MTF COVERAGE TEST")
print("=" * 78)

print(f"Original executable signals : {int(signals['XAUUSD'][0].sum() + signals['XAUUSD'][1].sum())}")
print(f"Strict MTF signals retained  : {kept}")
print(f"Signals removed              : {removed}")

periods = [
    ("2022-07-05", "M15 era"),
    ("2025-04-30", "M15 + M5 era"),
    ("2026-06-22", "FULL MTF era"),
]

for start_text, label in periods:
    start = pd.Timestamp(start_text, tz="UTC")
    mask = h1.index >= start

    lg = strict_long.copy()
    sh = strict_short.copy()

    lg[~mask] = False
    sh[~mask] = False

    market = MarketData(
        {"XAUUSD": h1},
        htf_hours=CFG.htf_hours,
        htf_offset=CFG.htf_offset_hours,
    )

    result = BacktestEngine(
        market,
        CFG,
        signal_override={"XAUUSD": (lg, sh)},
    ).run()

    final_equity = float(result.equity.iloc[-1])
    pnl = final_equity - result.initial_balance

    trades = pd.DataFrame(result.trades)

    print()
    print("-" * 78)
    print(label)
    print(f"Start          : {start}")
    print(f"Final equity   : ${final_equity:,.2f}")
    print(f"Net P/L        : ${pnl:,.2f}")
    print(f"Return         : {(final_equity/result.initial_balance-1)*100:.2f}%")
    print(f"Trades         : {len(trades)}")
    print(f"Halted         : {result.halted}")
    print(f"Halted at      : {result.halted_at}")

    if not trades.empty:
        p = pd.to_numeric(trades["pnl"], errors="coerce")
        gross_win = p[p > 0].sum()
        gross_loss = abs(p[p < 0].sum())

        print(f"Wins           : {(p > 0).sum()}")
        print(f"Losses         : {(p < 0).sum()}")
        print(f"Win rate       : {(p > 0).mean()*100:.2f}%")
        print(
            f"Profit factor  : "
            f"{gross_win/gross_loss if gross_loss else float('inf'):.3f}"
        )

        print("Sides:")
        print(trades["side"].value_counts().to_string())

        print("Exits:")
        print(trades["exit_reason"].value_counts().to_string())
    else:
        print("NO TRADES")

print()
print("=" * 78)
print("STRICT MTF TEST COMPLETE")
print("=" * 78)
