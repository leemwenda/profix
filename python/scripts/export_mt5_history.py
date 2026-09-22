"""Export H1 history + symbol specs from a running MT5 terminal (Windows, `pip install MetaTrader5`).
   python scripts/export_mt5_history.py --symbols EURUSD GBPUSD USDJPY AUDUSD USDCAD --years 6 --out data
The CSV 'time' column is SERVER time (MT5 convention). Pass --server-tz when loading (DST!).
"""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--symbols", nargs="+", required=True)
ap.add_argument("--years", type=float, default=6)
ap.add_argument("--out", default="data")
a = ap.parse_args()
import MetaTrader5 as mt5  # noqa: E402

if not mt5.initialize():
    raise SystemExit(f"initialize failed: {mt5.last_error()}")
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
specs = {}
end = datetime.now(timezone.utc); start = end - timedelta(days=365.25 * a.years)
for s in a.symbols:
    mt5.symbol_select(s, True)
    r = mt5.copy_rates_range(s, mt5.TIMEFRAME_H1, start, end)
    if r is None or len(r) == 0:
        print(f"{s}: no data ({mt5.last_error()})"); continue
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s")     # server-time wall clock
    df[["time", "open", "high", "low", "close", "tick_volume", "spread"]].to_csv(out / f"{s}.csv", index=False)
    i = mt5.symbol_info(s)
    specs[s] = {"digits": i.digits, "contract_size": i.trade_contract_size, "vol_min": i.volume_min, "vol_max": i.volume_max,
                "vol_step": i.volume_step, "stops_level": i.trade_stops_level, "tick_size": i.trade_tick_size,
                "tick_value_loss": i.trade_tick_value_loss, "swap_long": i.swap_long, "swap_short": i.swap_short}
    print(f"{s}: {len(df)} bars")
(out / "symbol_specs.json").write_text(json.dumps(specs, indent=2))
mt5.shutdown()
