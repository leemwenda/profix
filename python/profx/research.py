"""Research methodology: split -> select (IS) -> confirm (VAL) -> test ONCE (OOS).

Rules enforced in code, not just in documentation:
  * Parameters are searched on IN-SAMPLE only and confirmed on VALIDATION.
  * The final OOS window is evaluated exactly once per research state file (OOSGuard).
  * Selection objective is the t-statistic of per-trade R subject to trade-count,
    profit-factor and drawdown constraints - NOT net profit.
  * The verdict never says "profitable"; it says REJECT / INCONCLUSIVE / CANDIDATE.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ConfigError, RunConfig
from .engine import MarketData
from .metrics import compute_metrics, monthly_returns
from .robustness import (Window, buy_and_hold, concentration_flags, ema_cross_benchmark, monte_carlo,
                         parameter_sensitivity, random_entry_control, regime_breakdown, run_cfg, stress_test)

DEFAULT_GRID = {
    "donchian_n": [18, 24, 36],
    "sl_atr_mult": [1.25, 1.5, 2.0],
    "tp_r": [2.0, 3.0, 4.0],
    "trail_atr_mult": [1.5, 2.0, 3.0],
}
SENS_GRID = {
    "donchian_n": [12, 18, 24, 36, 48],
    "sl_atr_mult": [1.0, 1.25, 1.5, 2.0, 2.5],
    "tp_r": [1.5, 2.0, 3.0, 4.0, 5.0],
    "breakout_buffer_atr": [0.0, 0.05, 0.1, 0.2],
    "min_clv": [0.5, 0.6, 0.7],
}


class OOSAlreadyUsed(RuntimeError):
    pass


class OOSGuard:
    """Refuses a second evaluation of the OOS window (the classic way OOS data gets 'burned')."""

    def __init__(self, state_path: str | Path):
        self.path = Path(state_path)

    def used(self) -> bool:
        return self.path.exists()

    def run_once(self, cfg: RunConfig, fn, force: bool = False):
        if self.used() and not force:
            raise OOSAlreadyUsed(
                f"OOS window already evaluated ({self.path}). Re-testing after looking at OOS results "
                "invalidates it. Collect NEW data (or forward-test on demo) instead. force=True voids the OOS."
            )
        out = fn()
        self.path.write_text(json.dumps({
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": hashlib.sha256(repr(cfg).encode()).hexdigest(),
            "forced": bool(force and self.used()),
        }, indent=2))
        return out


def make_splits(market: MarketData, fractions=(0.5, 0.25, 0.25), embargo_days: int = 7) -> dict[str, Window]:
    if abs(sum(fractions) - 1.0) > 1e-9 or len(fractions) != 3:
        raise ConfigError("fractions must be 3 numbers summing to 1")
    T = pd.to_datetime(market.timeline, unit="ns", utc=True)
    n = len(T)
    a, b = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    emb = pd.Timedelta(days=embargo_days)
    return {
        "in_sample": (T[0], T[a]),
        "validation": (T[a] + emb, T[b]),
        "out_of_sample": (T[b] + emb, T[-1] + pd.Timedelta(hours=1)),
        "development": (T[0], T[b]),          # IS+VAL, used for everything except the final OOS test
    }


def robust_score(m: dict, min_trades: int = 60, dd_limit: float = 25.0) -> float:
    """Selection objective: t-stat of trade R with hard constraints (see module docstring)."""
    if m["total_trades"] < min_trades or not np.isfinite(m.get("profit_factor", np.nan)):
        return -math.inf
    if m["profit_factor"] <= 1.0 or m["max_drawdown_pct"] > dd_limit:
        return -math.inf
    return float(m["tstat_r"])


def grid_search(market: MarketData, cfg: RunConfig, grid: dict[str, list], window: Window,
                min_trades: int = 60, dd_limit: float = 25.0) -> pd.DataFrame:
    names = list(grid)
    rows = []
    for combo in itertools.product(*grid.values()):
        kw = dict(zip(names, combo))
        try:
            c = cfg.with_strategy(**kw)
        except ConfigError:
            continue
        _, m = run_cfg(market, c, window)
        rows.append({**kw, "score": robust_score(m, min_trades, dd_limit), "trades": m["total_trades"],
                     "pf": m["profit_factor"], "exp_r": m["expectancy_r"], "tstat": m["tstat_r"],
                     "ret_pct": m["return_pct"], "dd_pct": m["max_drawdown_pct"]})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def neighbour_plateau(df: pd.DataFrame, grid: dict[str, list]) -> dict:
    """Median score of grid neighbours (+-1 step in ONE parameter) relative to the best cell."""
    if df.empty or not np.isfinite(df.score.iloc[0]):
        return {"applicable": False}
    best = df.iloc[0]
    vals = []
    for p, opts in grid.items():
        i = opts.index(best[p])
        for j in (i - 1, i + 1):
            if 0 <= j < len(opts):
                m = df
                for q in grid:
                    m = m[m[q] == (opts[j] if q == p else best[q])]
                if len(m):
                    vals.append(m.score.iloc[0])
    fin = [v for v in vals if np.isfinite(v)]
    return {"applicable": True, "best_score": float(best.score), "n_neighbours": len(vals),
            "frac_neighbours_valid": len(fin) / len(vals) if vals else float("nan"),
            "median_neighbour_score": float(np.median(fin)) if fin else float("-inf"),
            "narrow_peak_flag": bool(not fin or len(fin) / len(vals) < 0.6 or np.median(fin) < 0.5 * best.score)}


def walk_forward(market: MarketData, cfg: RunConfig, grid: dict[str, list], within: Window,
                 train_days: int = 730, test_days: int = 180, step_days: int = 180,
                 min_trades: int = 40, dd_limit: float = 25.0) -> dict:
    lo, hi = pd.Timestamp(within[0]), pd.Timestamp(within[1])
    folds, oos_r, oos_pnl = [], [], []
    start = lo
    while start + pd.Timedelta(days=train_days + test_days) <= hi:
        tr = (start, start + pd.Timedelta(days=train_days))
        te = (tr[1], tr[1] + pd.Timedelta(days=test_days))
        g = grid_search(market, cfg, grid, tr, min_trades, dd_limit)
        best = g.iloc[0]
        chosen = {k: best[k] for k in grid}
        chosen = {k: (int(v) if float(v).is_integer() and k.endswith("_n") else float(v)) for k, v in chosen.items()}
        res, m = run_cfg(market, cfg.with_strategy(**chosen), te)
        folds.append({"train_start": tr[0].date(), "test_start": te[0].date(), "test_end": te[1].date(), **chosen,
                      "is_valid_choice": bool(np.isfinite(best.score)), "is_exp_r": float(best.exp_r),
                      "oos_trades": m["total_trades"], "oos_exp_r": m["expectancy_r"], "oos_pf": m["profit_factor"],
                      "oos_ret_pct": m["return_pct"]})
        if len(res.trades):
            oos_r += res.trades.r.tolist()
            oos_pnl += res.trades.pnl.tolist()
        start += pd.Timedelta(days=step_days)
    f = pd.DataFrame(folds)
    if f.empty:
        return {"folds": f, "note": "window too short for the requested train/test lengths"}
    r = np.array(oos_r)
    valid_is = f[f.is_valid_choice]
    wfe = (f.oos_exp_r.mean() / valid_is.is_exp_r.mean()) if len(valid_is) and valid_is.is_exp_r.mean() > 0 else float("nan")
    gp, gl = sum(x for x in oos_pnl if x > 0), -sum(x for x in oos_pnl if x <= 0)
    return {"folds": f, "n_folds": len(f), "pct_folds_profitable": float((f.oos_ret_pct > 0).mean()),
            "walk_forward_efficiency": float(wfe), "stitched_trades": int(len(r)),
            "stitched_expectancy_r": float(r.mean()) if len(r) else 0.0,
            "stitched_profit_factor": float(gp / gl) if gl > 0 else float("inf"),
            "stitched_r": r.tolist()}


def _summ(m: dict) -> dict:
    keys = ["total_trades", "net_profit", "return_pct", "profit_factor", "expectancy_r", "tstat_r", "win_rate",
            "max_drawdown_pct", "sharpe", "sortino", "calmar", "max_consecutive_losses", "exposure", "halted"]
    return {k: m.get(k) for k in keys}


def run_research(frames: dict[str, pd.DataFrame], cfg: RunConfig, out_dir: str | Path,
                 grid: dict[str, list] | None = None, sens_grid: dict[str, list] | None = None,
                 n_random: int = 60, mc_sims: int = 3000, seed: int = 0, fractions=(0.5, 0.25, 0.25),
                 top_k: int = 5, force_oos: bool = False, data_label: str = "unspecified",
                 min_trades_is: int = 60, wf_train_days: int = 730, wf_test_days: int = 180) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid, sens_grid = grid or DEFAULT_GRID, sens_grid or SENS_GRID
    market = MarketData(frames, cfg.htf_hours, cfg.htf_offset_hours)
    sp = make_splits(market, fractions)
    rep: dict = {"data_label": data_label, "symbols": list(market.symbols), "splits": {k: [str(v[0]), str(v[1])] for k, v in sp.items()},
                 "base_config": asdict(cfg), "grid": grid, "n_grid": int(np.prod([len(v) for v in grid.values()]))}

    # 1) select on IN-SAMPLE
    is_grid = grid_search(market, cfg, grid, sp["in_sample"], min_trades_is)
    is_grid.to_csv(out / "grid_in_sample.csv", index=False)
    rep["plateau_in_sample"] = neighbour_plateau(is_grid, grid)
    rep["multiple_testing"] = {"n_trials": rep["n_grid"], "null_max_tstat_approx": math.sqrt(2 * math.log(max(rep["n_grid"], 2))),
                               "best_is_tstat": float(is_grid.tstat.iloc[0])}

    # 2) confirm on VALIDATION: candidates = pre-registered default + top-K IS
    cands: list[tuple[str, dict]] = [("default", {})]
    for i in range(min(top_k, len(is_grid))):
        if np.isfinite(is_grid.score.iloc[i]):
            cands.append((f"is_rank_{i + 1}", {k: (int(is_grid[k].iloc[i]) if k.endswith("_n") else float(is_grid[k].iloc[i])) for k in grid}))
    val_rows = []
    for name, kw in cands:
        _, m = run_cfg(market, cfg.with_strategy(**kw), sp["validation"])
        val_rows.append({"candidate": name, **kw, "score": robust_score(m, 30), **_summ(m)})
    val = pd.DataFrame(val_rows).sort_values("score", ascending=False).reset_index(drop=True)
    val.to_csv(out / "validation_candidates.csv", index=False)
    chosen_name = val.candidate.iloc[0]
    chosen_kw = dict(next(kw for nme, kw in cands if nme == chosen_name))
    final_cfg = cfg.with_strategy(**chosen_kw)
    rep["chosen"] = {"name": chosen_name, "params": chosen_kw,
                     "note": "chosen on VALIDATION score; if no candidate passes constraints the pre-registered default is used"}

    # 3) development-window analytics (IS+VAL) for the chosen config
    dev_res, dev_m = run_cfg(market, final_cfg, sp["development"])
    rep["development"] = _summ(dev_m)
    sens_df, sens_v = parameter_sensitivity(market, final_cfg, {k: v for k, v in sens_grid.items()}, sp["development"])
    sens_df.to_csv(out / "sensitivity.csv", index=False)
    rep["sensitivity"] = sens_v
    stress = stress_test(market, final_cfg, sp["development"])
    stress.to_csv(out / "stress_development.csv", index=False)
    rep["stress_development"] = stress.to_dict("records")
    rep["random_entry_control"] = random_entry_control(market, final_cfg, dev_m, n_random, seed, sp["development"])
    rep["benchmark_ema_cross"] = _summ(ema_cross_benchmark(market, final_cfg, sp["development"]))
    rep["buy_and_hold"] = buy_and_hold(market, sp["development"])
    rep["mc_development"] = monte_carlo(dev_res.trades.r.to_numpy() if len(dev_res.trades) else np.array([]),
                                        final_cfg.risk.risk_pct, mc_sims, "bootstrap", seed)
    wf = walk_forward(market, cfg, grid, sp["development"], wf_train_days, wf_test_days, wf_test_days, 40)
    if isinstance(wf.get("folds"), pd.DataFrame) and len(wf["folds"]):
        wf["folds"].to_csv(out / "walk_forward_folds.csv", index=False)
    rep["walk_forward"] = {k: v for k, v in wf.items() if k not in ("folds", "stitched_r")}
    if "stitched_r" in wf and len(wf["stitched_r"]) >= 10:
        rep["walk_forward"]["mc_shuffle"] = monte_carlo(np.array(wf["stitched_r"]), cfg.risk.risk_pct, mc_sims, "shuffle", seed)
    rep["regimes_development"] = {k: v.reset_index().to_dict("records") for k, v in regime_breakdown(dev_res.trades).items()}
    rep["concentration_development"] = concentration_flags(dev_res.trades)

    # 4) OOS - exactly once
    guard = OOSGuard(out / "oos_state.json")

    def _oos():
        res, m = run_cfg(market, final_cfg, sp["out_of_sample"])
        _, harsh = run_cfg(market, final_cfg.with_costs(spread_mult=2.0, slippage_mult=3.0, commission_mult=1.5, exec_delay_seconds=10.0),
                           sp["out_of_sample"])
        return res, m, harsh

    res, oos_m, harsh = guard.run_once(final_cfg, _oos, force=force_oos)
    res.trades.to_csv(out / "oos_trades.csv", index=False)
    res.signals.to_csv(out / "oos_signal_log.csv", index=False)
    mr = monthly_returns(res.equity, res.initial_balance)
    mr.to_csv(out / "oos_monthly_returns.csv")
    rep["out_of_sample"] = _summ(oos_m)
    rep["out_of_sample_harsh_costs"] = _summ(harsh)
    rep["mc_oos"] = monte_carlo(res.trades.r.to_numpy() if len(res.trades) else np.array([]), final_cfg.risk.risk_pct, mc_sims, "bootstrap", seed + 1)
    rep["regimes_oos"] = {k: v.reset_index().to_dict("records") for k, v in regime_breakdown(res.trades).items()}
    rep["oos_rejections"] = res.rejections
    rep["verdict"] = verdict(rep)
    (out / "research_report.json").write_text(json.dumps(rep, indent=2, default=_json_default))
    (out / "research_report.md").write_text(render_markdown(rep))
    return rep


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


def verdict(rep: dict) -> dict:
    hard, soft = [], []
    o, dev, h = rep["out_of_sample"], rep["development"], rep["out_of_sample_harsh_costs"]
    n = o["total_trades"]
    if n < 100:
        soft.append(f"OOS has only {n} trades (<100): statistically weak")
    if not (o["expectancy_r"] > 0 and (o["profit_factor"] or 0) > 1.0):
        hard.append("OOS expectancy/profit factor is not positive")
    if dev["expectancy_r"] > 0 and o["expectancy_r"] < 0.4 * dev["expectancy_r"]:
        soft.append("OOS expectancy < 40% of development expectancy (degradation / possible overfit)")
    if rep["plateau_in_sample"].get("narrow_peak_flag"):
        soft.append("in-sample optimum is a narrow peak in the parameter grid")
    if rep["sensitivity"].get("narrow_peak_flag"):
        soft.append("performance collapses for neighbouring parameter values")
    mt = rep["multiple_testing"]
    if mt["best_is_tstat"] < mt["null_max_tstat_approx"]:
        soft.append("best in-sample t-stat is within what pure noise produces across the tested grid")
    if (h["profit_factor"] or 0) < 1.0:
        soft.append("not profitable under harsh execution costs (fragile to spread/slippage)")
    rc = rep["random_entry_control"]
    if rc["p_value_expectancy"] > 0.20:
        hard.append(f"random-entry control not beaten (p={rc['p_value_expectancy']:.2f}): no demonstrated entry edge")
    elif rc["p_value_expectancy"] > 0.05:
        soft.append(f"weak evidence vs random entries (p={rc['p_value_expectancy']:.2f})")
    wf = rep["walk_forward"]
    if wf.get("n_folds", 0) and (wf.get("pct_folds_profitable", 0) < 0.6 or not (wf.get("walk_forward_efficiency", 0) >= 0.5)):
        soft.append("walk-forward: <60% profitable folds or efficiency <0.5")
    mc = rep["mc_oos"]
    if mc.get("prob_loss", 0) > 0.3:
        soft.append(f"Monte Carlo: {mc['prob_loss']:.0%} of bootstrap paths lose money")
    c = rep["concentration_development"]
    if c.get("flag_few_trades_dominate") or c.get("flag_single_year_dominates"):
        soft.append("profits concentrated in few trades or a single year")
    if o.get("halted"):
        soft.append(f"risk breaker tripped in OOS ({o['halted']})")
    status = "REJECT" if hard else ("INCONCLUSIVE" if soft else "CANDIDATE_FOR_DEMO_FORWARD_TEST")
    return {"status": status, "hard_failures": hard, "warnings": soft,
            "disclaimer": "No verdict here implies future profitability. Passing only earns the right to a demo forward test."}


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def render_markdown(rep: dict) -> str:
    L = [f"# ProFX research report\n", f"Data: **{rep['data_label']}**  |  symbols: {', '.join(rep['symbols'])}\n"]
    if "synthetic" in rep["data_label"].lower():
        L.append("> **SYNTHETIC DATA - pipeline demonstration only. These numbers say nothing about real markets.**\n")
    v = rep["verdict"]
    L += [f"## Verdict: `{v['status']}`\n", v["disclaimer"] + "\n"]
    for x in v["hard_failures"]:
        L.append(f"- **FAIL:** {x}")
    for x in v["warnings"]:
        L.append(f"- warning: {x}")
    L.append("\n## Data splits (chronological, embargo between segments)\n")
    for k, (a, b) in rep["splits"].items():
        L.append(f"- {k}: {a[:10]} -> {b[:10]}")
    L.append(f"\n## Selection\n\nGrid size {rep['n_grid']}. Chosen: `{rep['chosen']['name']}` {rep['chosen']['params']}. "
             f"Best IS t-stat {_fmt(rep['multiple_testing']['best_is_tstat'], 2)} vs noise-max approx "
             f"{_fmt(rep['multiple_testing']['null_max_tstat_approx'], 2)}.\n")
    hdr = "| window | trades | ret % | PF | E[R] | t | win | maxDD % | Sharpe | Sortino |\n|---|---|---|---|---|---|---|---|---|---|"
    L += ["## Results\n", hdr]
    for name, key in (("development (IS+VAL)", "development"), ("OUT-OF-SAMPLE", "out_of_sample"), ("OOS harsh costs", "out_of_sample_harsh_costs")):
        m = rep[key]
        L.append(f"| {name} | {m['total_trades']} | {_fmt(m['return_pct'], 1)} | {_fmt(m['profit_factor'], 2)} | {_fmt(m['expectancy_r'])} | "
                 f"{_fmt(m['tstat_r'], 2)} | {_fmt(m['win_rate'], 2)} | {_fmt(m['max_drawdown_pct'], 1)} | {_fmt(m['sharpe'], 2)} | {_fmt(m['sortino'], 2)} |")
    rc, b = rep["random_entry_control"], rep["benchmark_ema_cross"]
    L += ["\n## Controls\n",
          f"- Random entries ({rc['n_runs']} runs): E[R] median {_fmt(rc['random_expectancy_r']['p50'])}, p95 {_fmt(rc['random_expectancy_r']['p95'])}; "
          f"strategy {_fmt(rc['strategy_expectancy_r'])}; p-value {_fmt(rc['p_value_expectancy'], 3)}",
          f"- EMA(20/50) cross benchmark (same exits/risk): trades {b['total_trades']}, PF {_fmt(b['profit_factor'], 2)}, E[R] {_fmt(b['expectancy_r'])}",
          f"- Buy & hold basket (price only): {_fmt(rep['buy_and_hold'].get('basket_return_pct'), 1)}% , maxDD {_fmt(rep['buy_and_hold'].get('basket_max_dd_pct'), 1)}%"]
    s = rep["sensitivity"]
    L += ["\n## Robustness\n",
          f"- Parameter neighbours >= 50% of base: {_fmt(s['frac_neighbours_ge_half_base'], 2)}; positive: {_fmt(s['frac_neighbours_positive'], 2)}; narrow-peak flag: {s['narrow_peak_flag']}",
          f"- Walk-forward: {rep['walk_forward'].get('n_folds', 0)} folds, profitable {_fmt(rep['walk_forward'].get('pct_folds_profitable'), 2)}, "
          f"efficiency {_fmt(rep['walk_forward'].get('walk_forward_efficiency'), 2)}, stitched E[R] {_fmt(rep['walk_forward'].get('stitched_expectancy_r'))}",
          f"- Monte Carlo (OOS bootstrap): P(loss) {_fmt(rep['mc_oos'].get('prob_loss'), 2)}, 95th pct maxDD {_fmt(rep['mc_oos'].get('max_dd_pct', {}).get('p95'), 1)}%"]
    L += ["\n## Cost stress (development window)\n", "| scenario | trades | PF | E[R] | ret % | maxDD % |", "|---|---|---|---|---|---|"]
    for r in rep["stress_development"]:
        L.append(f"| {r['scenario']} | {r['total_trades']} | {_fmt(r['profit_factor'], 2)} | {_fmt(r['expectancy_r'])} | {_fmt(r['return_pct'], 1)} | {_fmt(r['max_drawdown_pct'], 1)} |")
    L.append("\nFiles: grid_in_sample.csv, validation_candidates.csv, sensitivity.csv, stress_development.csv, walk_forward_folds.csv, "
             "oos_trades.csv, oos_signal_log.csv, oos_monthly_returns.csv, research_report.json\n")
    return "\n".join(L)
