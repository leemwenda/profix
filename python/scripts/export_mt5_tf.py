import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import MetaTrader5 as mt5

ap = argparse.ArgumentParser()
ap.add_argument("--symbol", default="XAUUSD")
ap.add_argument("--tfs", nargs="+", default=["M5"])
ap.add_argument("--start", default="2022-01-01")
ap.add_argument("--out", default="mtf_new")
a = ap.parse_args()

TF = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
      "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}

if not mt5.initialize():
    raise SystemExit(f"initialize failed: {mt5.last_error()}")
mt5.symbol_select(a.symbol, True)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
start = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
end = datetime.now(timezone.utc)

for name in a.tfs:
    r = mt5.copy_rates_range(a.symbol, TF[name], start, end)
    if r is None or len(r) == 0:
        print(f"{name}: no data ({mt5.last_error()})"); continue
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s").dt.strftime("%Y.%m.%d %H:%M")
    cols = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    df[cols].to_csv(out / f"{a.symbol}_{name}.csv", index=False, float_format="%.8f")
    print(f"{name}: {len(df)} bars, first {df['time'].iloc[0]}, last {df['time'].iloc[-1]}")
mt5.shutdown()
