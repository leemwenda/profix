from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profx.config import load_config
from profx.mtf import load_mtf
from profx.mtf_backtest import build_mtf_signals
from profx.engine import MarketData, BacktestEngine

ROOT = Path(__file__).resolve().parents[2]
CFG0 = load_config(ROOT / "config/xauusd.toml")
CFG = CFG0.with_risk(max_drawdown_pct=99.0)   # no halt, same for both
mtf = load_mtf(ROOT / "python/data/mtf", "XAUUSD")
WINDOW = (pd.Timestamp("2022-07-05", tz="UTC"), None)  # where M15 starts


def pf(p):
    gl = abs(p[p < 0].sum())
    return p[p > 0].sum() / gl if gl else float("inf")


def run(cfg, h1, signals=None, window=WINDOW):
    md = MarketData({"XAUUSD": h1}, htf_hours=cfg.htf_hours,
                    htf_offset=cfg.htf_offset_hours)
    res = BacktestEngine(md, cfg, signal_override=signals, window=window).run()
    eq = res.equity.astype(float)
    t = pd.DataFrame(res.trades)
    if t.empty:
        return dict(n=0, ret=0.0, pf=0.0, dd=0.0, yearly=pd.Series(dtype=float), halted=res.halted)
    t["pnl"] = pd.to_numeric(t["pnl"], errors="coerce")
    t["y"] = pd.to_datetime(t["entry_time"], utc=True).dt.year
    return dict(n=len(t), ret=(eq.iloc[-1] / res.initial_balance - 1) * 100,
                pf=pf(t["pnl"].dropna()),
                dd=((eq.cummax() - eq) / eq.cummax()).max() * 100,
                yearly=t.groupby("y")["pnl"].sum(), halted=res.halted)


# Sanity check: control with the ORIGINAL config, full history.
r0 = run(CFG0, mtf.frames["H1"].copy(), window=(None, None))
print(f"SANITY control, original config, all data: trades={r0['n']} "
      f"return={r0['ret']:.2f}% PF={r0['pf']:.3f} halted={r0['halted']}")
print("(earlier notes said 174 trades, +3.19%, PF 1.195)\n")

mtf_sig, _ = build_mtf_signals(mtf=mtf, cfg=CFG, symbol="XAUUSD")

rows, yearly = [], {}
for label, floor in (("native spread", None), ("spread floor 20", 20)):
    h1 = mtf.frames["H1"].copy()
    if floor is not None:
        h1["spread"] = np.maximum(h1["spread"].fillna(floor), floor)
    for name, sig in (("H1 control", None), ("MTF", mtf_sig)):
        r = run(CFG, h1, signals=sig)
        rows.append((f"{name} / {label}", r))
        yearly[f"{name} / {label}"] = r["yearly"]

print(f"{'run':<34}{'trades':>7}{'return%':>9}{'PF':>8}{'maxDD%':>8}")
for k, r in rows:
    print(f"{k:<34}{r['n']:>7}{r['ret']:>9.2f}{r['pf']:>8.3f}{r['dd']:>8.2f}")

print("\nNET P/L BY ENTRY YEAR")
print(pd.DataFrame(yearly).fillna(0).round(2).to_string())

h = mtf.frames["H1"]
h = h[h.index >= WINDOW[0]]
print(f"\nGold over the same dates: {h['close'].iloc[0]:.0f} -> {h['close'].iloc[-1]:.0f} "
      f"({(h['close'].iloc[-1] / h['close'].iloc[0] - 1) * 100:.0f}%)")
