from pathlib import Path
import pandas as pd
import numpy as np

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

print("=" * 80)
print("MTF CONFIRMATION DIAGNOSTIC")
print("=" * 80)

print(f"Decision records: {len(records):,}")

# Inspect the actual record structure first.
if records:
    r = records[0]
    print()
    print("Record type:")
    print(type(r))
    print()
    print("Record fields:")
    if hasattr(r, "__dataclass_fields__"):
        print(list(r.__dataclass_fields__.keys()))
    elif hasattr(r, "_fields"):
        print(list(r._fields))
    else:
        print(vars(r).keys() if hasattr(r, "__dict__") else dir(r))

# Convert dataclass records safely.
rows = []
for r in records:
    if hasattr(r, "__dataclass_fields__"):
        row = {
            name: getattr(r, name)
            for name in r.__dataclass_fields__
        }
    elif hasattr(r, "_asdict"):
        row = dict(r._asdict())
    else:
        row = vars(r)

    rows.append(row)

df = pd.DataFrame(rows)

print()
print("Available diagnostic columns:")
print(df.columns.tolist())

print()
print("=" * 80)
print("CONFIRMATION DISTRIBUTION")
print("=" * 80)

for col in ["m15_valid", "m5_valid", "m1_valid"]:
    if col in df.columns:
        print()
        print(col)
        print(df[col].value_counts(dropna=False).to_string())

if "setup_score" in df.columns:
    print()
    print("Setup score:")
    print(df["setup_score"].value_counts().sort_index().to_string())

if "decision_score" in df.columns:
    print()
    print("Decision score:")
    print(df["decision_score"].value_counts().sort_index().to_string())

print()
print("=" * 80)
print("COMBINATION COUNTS")
print("=" * 80)

needed = ["m15_valid", "m5_valid", "m1_valid"]

if all(c in df.columns for c in needed):
    combos = (
        df.groupby(needed)
        .size()
        .sort_values(ascending=False)
    )
    print(combos.to_string())

# ---------------------------------------------------------------
# Create several NON-PRODUCTION experiments.
#
# A) Existing strategy
# B) M5 confirmation required
# C) M5 + M1 confirmation required
#
# These are applied only where the corresponding confirmation
# fields exist in the diagnostic records.
# ---------------------------------------------------------------

if "h1_time" in df.columns:
    time_col = "h1_time"
elif "decision_time" in df.columns:
    time_col = "decision_time"
elif "timestamp" in df.columns:
    time_col = "timestamp"
else:
    time_col = None

print()
print("=" * 80)
print("TIME FIELD")
print("=" * 80)
print(time_col)

if time_col:
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

# Build lookup by H1 decision time.
record_lookup = {}

if time_col:
    for _, row in df.iterrows():
        record_lookup[pd.Timestamp(row[time_col])] = row


def make_override(mode):
    long_arr = signals["XAUUSD"][0].copy()
    short_arr = signals["XAUUSD"][1].copy()

    removed = 0

    if not time_col:
        return long_arr, short_arr, removed

    for i, h1_open in enumerate(h1.index):
        decision_time = h1_open + pd.Timedelta(hours=1)

        row = record_lookup.get(decision_time)

        if row is None:
            continue

        m5 = bool(row.get("m5_valid", False))
        m1 = bool(row.get("m1_valid", False))

        keep = True

        if mode == "M5":
            keep = m5

        elif mode == "M5_M1":
            keep = m5 and m1

        if (long_arr[i] or short_arr[i]) and not keep:
            long_arr[i] = False
            short_arr[i] = False
            removed += 1

    return long_arr, short_arr, removed


experiments = [
    ("BASELINE CURRENT MTF", "BASE"),
    ("REQUIRE M5 CONFIRMATION", "M5"),
    ("REQUIRE M5 + M1 CONFIRMATION", "M5_M1"),
]

results = []

for label, mode in experiments:

    if mode == "BASE":
        lg = signals["XAUUSD"][0].copy()
        sh = signals["XAUUSD"][1].copy()
        removed = 0
    else:
        lg, sh, removed = make_override(mode)

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

    trades = pd.DataFrame(result.trades)

    final_equity = float(result.equity.iloc[-1])
    pnl = final_equity - result.initial_balance

    if not trades.empty:
        p = pd.to_numeric(trades["pnl"], errors="coerce")
        wins = int((p > 0).sum())
        losses = int((p < 0).sum())
        gross_win = float(p[p > 0].sum())
        gross_loss = abs(float(p[p < 0].sum()))
        pf = gross_win / gross_loss if gross_loss else float("inf")
        win_rate = wins / len(p) * 100
    else:
        wins = losses = 0
        pf = float("nan")
        win_rate = 0

    results.append({
        "experiment": label,
        "signals_removed": removed,
        "trades": len(trades),
        "final_equity": final_equity,
        "net_pnl": pnl,
        "return_pct": (final_equity / result.initial_balance - 1) * 100,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": pf,
        "halted": result.halted,
    })

print()
print("=" * 80)
print("BACKTEST COMPARISON")
print("=" * 80)

out = pd.DataFrame(results)

print(
    out.to_string(
        index=False,
        formatters={
            "final_equity": lambda x: f"${x:,.2f}",
            "net_pnl": lambda x: f"${x:,.2f}",
            "return_pct": lambda x: f"{x:.2f}%",
            "win_rate": lambda x: f"{x:.2f}%",
            "profit_factor": lambda x: f"{x:.3f}",
        },
    )
)

print()
print("=" * 80)
print("TEST COMPLETE — NO PRODUCTION FILES MODIFIED")
print("=" * 80)
