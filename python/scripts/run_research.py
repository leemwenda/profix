"""Full research pipeline (IS -> VAL -> OOS once, walk-forward, MC, stress, controls).
   python scripts/run_research.py --data-dir data --server-tz Europe/Athens --out out/research
   python scripts/run_research.py --synthetic 6 --out out/research_demo      # pipeline demo
"""
import argparse
from pathlib import Path

import _common  # noqa: F401
from _common import get_frames, load_config

from profx.research import run_research

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "config" / "strategy.toml"))
ap.add_argument("--data-dir"), ap.add_argument("--synthetic", type=float)
ap.add_argument("--server-tz"), ap.add_argument("--server-offset", type=float)
ap.add_argument("--out", default="out/research")
ap.add_argument("--random-runs", type=int, default=100), ap.add_argument("--mc-sims", type=int, default=5000)
ap.add_argument("--force-oos", action="store_true", help="VOIDS the out-of-sample test (recorded in oos_state.json)")
a = ap.parse_args()
cfg = load_config(a.config).with_risk(dd_halt_cooldown_days=30)   # research: simulate operator reset (see docs)
frames, label = get_frames(cfg, a.data_dir, a.synthetic, a.server_tz, a.server_offset)
rep = run_research(frames, cfg, a.out, n_random=a.random_runs, mc_sims=a.mc_sims, force_oos=a.force_oos, data_label=label)
v = rep["verdict"]
print(f"VERDICT: {v['status']}")
for x in v["hard_failures"]:
    print("  FAIL   :", x)
for x in v["warnings"]:
    print("  warning:", x)
print(f"\nFull report: {Path(a.out) / 'research_report.md'}")
