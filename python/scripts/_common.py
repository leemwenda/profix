import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profx.config import RunConfig, load_config  # noqa: E402
from profx.data import load_csv, synthetic_fx  # noqa: E402


def get_frames(cfg: RunConfig, data_dir, synthetic_years, server_tz, server_offset, seed=1):
    if synthetic_years:
        return {s: synthetic_fx(s, years=synthetic_years, seed=seed) for s in cfg.symbols}, f"SYNTHETIC ({synthetic_years}y, seed {seed})"
    if not data_dir:
        raise SystemExit("Provide --data-dir (files <SYMBOL>.csv) or --synthetic YEARS")
    frames = {}
    for s in cfg.symbols:
        p = Path(data_dir) / f"{s}.csv"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        frames[s] = load_csv(p, server_tz=server_tz, server_utc_offset_hours=server_offset)
    return frames, f"CSV {data_dir}"
