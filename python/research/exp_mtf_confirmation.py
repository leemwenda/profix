"""
M5/M1 confirmation ablation on ONE common eligible window.

A: M15 mandatory, M5/M1 optional   (current behaviour)
B: M15 + M5 mandatory, M1 optional
C: M15 + M5 + M1 mandatory

Stored decision.setup_score is on a 0-50 scale: round(raw / 30 * 50)
    raw  5 ->  8   M1 only
    raw 10 -> 17   M5 only
    raw 15 -> 25   M15 only
    raw 20 -> 33   M15 + M1
    raw 25 -> 42   M15 + M5
    raw 30 -> 50   M15 + M5 + M1
"""
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
CFG = CFG0.with_risk(max_drawdown_pct=99.0)      # identical, no halt, for all variants
SYMBOL = "XAUUSD"

M15_SET = {25, 33, 42, 50}
M5_SET = {17, 42, 50}
M1_SET = {8, 33, 50}

WIN_START = pd.Timestamp("2025-05-02", tz="UTC")  # first day with M5 coverage

mtf = load_mtf(ROOT / "python/data/mtf", SYMBOL)
h1 = mtf.frames["H1"].copy()

signals, records = build_mtf_signals(mtf=mtf, cfg=CFG, symbol=SYMBOL)
long_a, short_a = signals[SYMBOL]

assert len(records) == len(long_a) == len(h1), "record/H1 length mismatch"

ss = np.array([r.decision.setup_score for r in records])
dt = pd.DatetimeIndex([pd.Timestamp(r.decision_time) for r in records])
if dt.tz is None:
    dt = dt.tz_localize("UTC")
tier_by_time = {pd.Timestamp(r.decision_time): r.decision.tier for r in records}
score_by_time = {pd.Timestamp(r.decision_time): r.decision.score for r in records}

elig = np.asarray(dt >= WIN_START)
has_m15 = np.isin(ss, list(M15_SET))
has_m5 = np.isin(ss, list(M5_SET))
has_m1 = np.isin(ss, list(M1_SET))

# Sanity: every executable signal must have M15 (M15 mandatory)
exe = long_a | short_a
print("Executable signals total          :", int(exe.sum()))
print("Executable without M15 (must be 0):", int((exe & ~has_m15).sum()))

variants = {
    "A_M15_ONLY_OK":    elig,
    "B_M5_REQUIRED":    elig & has_m5,
    "C_M5_M1_REQUIRED": elig & has_m5 & has_m1,
}


def pf(p):
    gl = abs(p[p < 0].sum())
    return float(p[p > 0].sum() / gl) if gl else float("inf")


def dd_duration(eq):
    try:
        under = eq < eq.cummax()
        grp = (~under).cumsum()
        spans = [(g.index.max() - g.index.min()) for _, g in eq[under].groupby(grp[under])]
        return max(spans) if spans else pd.Timedelta(0)
    except Exception:
        return None


def run_variant(name, mask):
    lg = long_a & mask
    sh = short_a & mask
    n_sig = int(lg.sum() + sh.sum())

    md = MarketData({SYMBOL: h1}, htf_hours=CFG.htf_hours, htf_offset=CFG.htf_offset_hours)
    res = BacktestEngine(md, CFG, signal_override={SYMBOL: (lg, sh)},
                         window=(WIN_START, None)).run()

    eq = res.equity.astype(float)
    final = float(eq.iloc[-1])
    t = pd.DataFrame(res.trades)

    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)
    print(f"Signals            : {n_sig}")
    print(f"Trades             : {len(t)}")
    print(f"Final equity       : ${final:,.2f}")
    print(f"Net P/L            : ${final - res.initial_balance:,.2f}")
    print(f"Return             : {(final / res.initial_balance - 1) * 100:.2f}%")
    print(f"Max DD             : ${(eq.cummax() - eq).max():,.2f} "
          f"({((eq.cummax() - eq) / eq.cummax()).max() * 100:.2f}%)")
    print(f"Longest DD duration: {dd_duration(eq)}")
    print(f"Halted             : {res.halted}")
    print(f"Rejections         : {getattr(res, 'rejections', None)}")

    out = dict(name=name, sig=n_sig, n=len(t), net=final - res.initial_balance)
    if t.empty:
        return out

    t["pnl"] = pd.to_numeric(t["pnl"], errors="coerce")
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    t["hours"] = (t["exit_time"] - t["entry_time"]).dt.total_seconds() / 3600
    t["y"] = t["entry_time"].dt.year
    p = t["pnl"].dropna()

    print(f"Profit factor      : {pf(p):.3f}")
    print(f"Win rate           : {(p > 0).mean() * 100:.2f}%")
    print(f"Expectancy/avg     : ${p.mean():.2f}")
    print(f"Median trade       : ${p.median():.2f}")
    print(f"Avg hold (hours)   : {t['hours'].mean():.2f}")
    print(f"Trades per week    : {len(t) / max(1, (t['entry_time'].max() - t['entry_time'].min()).days / 7):.2f}")

    # t-stat of per-trade P/L (rough, not proof)
    if len(p) > 2 and p.std() > 0:
        print(f"t-stat (mean pnl)  : {p.mean() / (p.std() / np.sqrt(len(p))):.2f}")

    print("\nBy side:")
    print(t.groupby("side")["pnl"].agg(trades="size", net="sum", pf=pf, avg="mean").round(2).to_string())

    print("\nBy entry year:")
    print(t.groupby("y")["pnl"].agg(trades="size", net="sum", pf=pf, avg="mean").round(2).to_string())

    print("\nExit reasons:")
    print(t["exit_reason"].value_counts().to_string())

    t["tier"] = t["entry_time"].map(tier_by_time)
    t["score"] = t["entry_time"].map(score_by_time)
    print(f"\nTrades unmatched to decision record: {int(t['tier'].isna().sum())}")
    print("By decision tier:")
    print(t.groupby("tier")["pnl"].agg(trades="size", net="sum", pf=pf, avg="mean").round(2).to_string())

    out.update(pf=pf(p), win=(p > 0).mean() * 100, avg=p.mean(),
               ret=(final / res.initial_balance - 1) * 100)
    return out


print("\nWindow start:", WIN_START, "| config: max_drawdown_pct=99 (no halt)")
rows = [run_variant(n, m) for n, m in variants.items()]

print("\n" + "=" * 100)
print("SUMMARY (same window for all variants)")
print("=" * 100)
print(f"{'variant':<20}{'signals':>8}{'trades':>8}{'net$':>10}{'ret%':>8}{'PF':>8}{'win%':>8}{'avg$':>8}")
for r in rows:
    print(f"{r['name']:<20}{r['sig']:>8}{r['n']:>8}{r['net']:>10.2f}"
          f"{r.get('ret', 0):>8.2f}{r.get('pf', float('nan')):>8.3f}"
          f"{r.get('win', 0):>8.2f}{r.get('avg', 0):>8.2f}")

print("""
READ THIS BEFORE CONCLUDING:
- Variants share dates, so differences are not regime differences, but they are
  still one path with overlapping trades. Engine path-dependence (cooldown,
  daily limits) means B is not simply "A minus some trades".
- M5 confirmation is correlated with H1 regime strength; this script does not
  separate the two (see exp_m5_incremental.py).
- C (M1 required) will have very few trades. Do not draw conclusions from it.
No production files were modified.
""")
