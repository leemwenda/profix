"""Single backtest with full metrics.  Example:
   python scripts/run_backtest.py --config ../config/strategy.toml --synthetic 4 --out out/bt
   python scripts/run_backtest.py --config ../config/strategy.toml --data-dir data --server-tz Europe/Athens
"""
import argparse
import json
from pathlib import Path

import _common  # noqa: F401
from _common import get_frames, load_config

from profx.engine import BacktestEngine, MarketData
from profx.metrics import compute_metrics, format_metrics, monthly_returns

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "config" / "strategy.toml"))
ap.add_argument("--data-dir"), ap.add_argument("--synthetic", type=float)
ap.add_argument("--server-tz"), ap.add_argument("--server-offset", type=float)
ap.add_argument("--start"), ap.add_argument("--end")
ap.add_argument("--out", default="out/backtest")
ap.add_argument("--plot", action="store_true")
a = ap.parse_args()

cfg = load_config(a.config)
frames, label = get_frames(cfg, a.data_dir, a.synthetic, a.server_tz, a.server_offset)
md = MarketData(frames, cfg.htf_hours, cfg.htf_offset_hours)
res = BacktestEngine(md, cfg, window=(a.start, a.end)).run()
m = compute_metrics(res)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
res.trades.to_csv(out / "trades.csv", index=False)
res.signals.to_csv(out / "signal_log.csv", index=False)
res.equity.to_csv(out / "equity.csv")
monthly_returns(res.equity, res.initial_balance).to_csv(out / "monthly_returns.csv")
(out / "metrics.json").write_text(json.dumps({**m, "rejections": res.rejections, "data": label}, indent=2, default=str))
print(f"Data: {label}\n")
print(format_metrics(m))
print("\nSignal rejections:", res.rejections)
print(f"\nMonthly returns (%):\n{monthly_returns(res.equity, res.initial_balance)}")
if a.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    res.equity.plot(ax=ax[0], title="Equity")
    (res.equity / res.equity.cummax() - 1).mul(100).plot(ax=ax[1], title="Drawdown %")
    fig.tight_layout(); fig.savefig(out / "equity.png", dpi=110)
if "SYNTHETIC" in label:
    print("\n*** SYNTHETIC DATA: numbers demonstrate the pipeline only ***")
