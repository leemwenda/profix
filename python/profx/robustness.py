"""Robustness / anti-overfitting toolkit.

Every function takes a MarketData + RunConfig and reuses the SAME engine as the main
backtest, so stress and control tests cannot drift from the real logic.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from .config import RunConfig
from .engine import BacktestEngine, BacktestResult, MarketData
from .metrics import compute_metrics, drawdown_stats
from .signals import entry_filter_mask

Window = tuple[pd.Timestamp | None, pd.Timestamp | None]
KEYS = ["total_trades", "profit_factor", "expectancy_r", "tstat_r", "return_pct", "max_drawdown_pct", "win_rate"]


def run_cfg(market: MarketData, cfg: RunConfig, window: Window = (None, None), **kw) -> tuple[BacktestResult, dict]:
    res = BacktestEngine(market, cfg, window=window, **kw).run()
    return res, compute_metrics(res)


# ------------------------------------------------------------------ Monte Carlo on trade sequence
def monte_carlo(r: np.ndarray, risk_pct: float, n_sims: int = 5000, mode: str = "bootstrap", seed: int = 0,
                dd_threshold_pct: float = 20.0, ruin_equity_frac: float = 0.5) -> dict:
    """Fixed-fractional compounding of per-trade R multiples.

    mode="shuffle":   random ORDER of the realised trades (final equity identical, DD varies)
    mode="bootstrap": resample WITH replacement (also tests dependence on specific trades)
    """
    r = np.asarray(r, float)
    n = len(r)
    if n < 10:
        return {"n_trades": n, "note": "too few trades for Monte Carlo"}
    rng = np.random.default_rng(seed)
    f = risk_pct / 100.0
    finals = np.empty(n_sims)
    dds = np.empty(n_sims)
    for k in range(n_sims):
        seq = rng.permutation(r) if mode == "shuffle" else rng.choice(r, size=n, replace=True)
        curve = np.cumprod(np.maximum(1.0 + seq * f, 1e-9))
        peak = np.maximum.accumulate(np.concatenate([[1.0], curve]))[1:]
        dds[k] = ((peak - curve) / peak).max() * 100.0
        finals[k] = (curve[-1] - 1.0) * 100.0
    q = lambda a, p: float(np.percentile(a, p))  # noqa: E731
    return {
        "mode": mode, "n_trades": n, "n_sims": n_sims,
        "final_return_pct": {"p5": q(finals, 5), "p50": q(finals, 50), "p95": q(finals, 95)},
        "max_dd_pct": {"p50": q(dds, 50), "p95": q(dds, 95), "p99": q(dds, 99)},
        "prob_loss": float((finals < 0).mean()),
        f"prob_dd_gt_{dd_threshold_pct:g}pct": float((dds > dd_threshold_pct).mean()),
        "prob_ruin": float(((finals / 100.0 + 1.0) < ruin_equity_frac).mean()),
    }


# ------------------------------------------------------------------ cost / execution stress
def stress_test(market: MarketData, cfg: RunConfig, window: Window = (None, None)) -> pd.DataFrame:
    scen = [
        ("baseline", {}),
        ("spread x1.5", dict(spread_mult=1.5)), ("spread x2", dict(spread_mult=2.0)), ("spread x3", dict(spread_mult=3.0)),
        ("slippage x2", dict(slippage_mult=2.0)), ("slippage x4", dict(slippage_mult=4.0)),
        ("commission x2", dict(commission_mult=2.0)),
        ("delay 5s", dict(exec_delay_seconds=5.0)), ("delay 30s", dict(exec_delay_seconds=30.0)),
        ("swap x3", dict(swap_mult=3.0)),
        ("COMBINED harsh (spr x2, slip x3, comm x1.5, delay 10s)",
         dict(spread_mult=2.0, slippage_mult=3.0, commission_mult=1.5, exec_delay_seconds=10.0)),
    ]
    rows = []
    for name, kw in scen:
        _, m = run_cfg(market, cfg.with_costs(**kw), window)
        rows.append({"scenario": name, **{k: m.get(k) for k in KEYS}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ parameter sensitivity
def parameter_sensitivity(market: MarketData, cfg: RunConfig, grid: dict[str, list], window: Window = (None, None),
                          metric: str = "expectancy_r") -> tuple[pd.DataFrame, dict]:
    """One-at-a-time sweeps around the base config + a 'plateau vs spike' verdict."""
    _, base = run_cfg(market, cfg, window)
    rows = []
    for p, values in grid.items():
        for v in values:
            _, m = run_cfg(market, cfg.with_strategy(**{p: v}), window)
            rows.append({"param": p, "value": v, "is_base": getattr(cfg.strategy, p) == v, **{k: m.get(k) for k in KEYS}})
    df = pd.DataFrame(rows)
    b = base.get(metric, float("nan"))
    nb = df[~df.is_base][metric].astype(float)
    frac_ok = float((nb >= 0.5 * b).mean()) if b > 0 and len(nb) else float("nan")
    frac_pos = float((nb > 0).mean()) if len(nb) else float("nan")
    verdict = {
        "base_metric": b, "neighbour_median": float(nb.median()) if len(nb) else float("nan"),
        "frac_neighbours_ge_half_base": frac_ok, "frac_neighbours_positive": frac_pos,
        "narrow_peak_flag": bool(b > 0 and (frac_ok < 0.6 or frac_pos < 0.7)),
    }
    return df, verdict


# ------------------------------------------------------------------ controls
def _strategy_signal_counts(market: MarketData, cfg: RunConfig, lo: int, hi: int) -> dict[str, int]:
    from .signals import make_signals
    out = {}
    for s in market.symbols:
        ind = market.indicators(s, cfg.strategy)
        lg, sh = make_signals(ind, cfg.strategy, market.specs[s])
        m = (ind.t >= lo) & (ind.t < hi)
        out[s] = int(((lg | sh) & m).sum())
    return out


def random_entry_control(market: MarketData, cfg: RunConfig, strategy_metrics: dict, n_runs: int = 100,
                         seed: int = 0, window: Window = (None, None)) -> dict:
    """Same exits/risk/filters/frequency, but RANDOM direction and RANDOM timing.

    If the real strategy is not clearly better than this distribution, the entry
    logic carries no demonstrable information (the exit/risk framework alone does not
    create an edge).
    """
    lo, hi = market.slice_ns(*window)
    counts = _strategy_signal_counts(market, cfg, lo, hi)
    rng = np.random.default_rng(seed)
    masks = {}
    for s in market.symbols:
        ind = market.indicators(s, cfg.strategy)
        masks[s] = np.where(entry_filter_mask(ind, cfg.strategy, market.specs[s]) & (ind.t >= lo) & (ind.t < hi))[0]
    exp_r, pf = [], []
    for _ in range(n_runs):
        ov = {}
        for s in market.symbols:
            n = len(market.t[s])
            lg, sh = np.zeros(n, bool), np.zeros(n, bool)
            k = min(counts[s], len(masks[s]))
            if k:
                pick = rng.choice(masks[s], size=k, replace=False)
                side = rng.random(k) < 0.5
                lg[pick[side]] = True
                sh[pick[~side]] = True
            ov[s] = (lg, sh)
        _, m = run_cfg(market, cfg, window, signal_override=ov)
        exp_r.append(m["expectancy_r"])
        pf.append(m["profit_factor"] if np.isfinite(m["profit_factor"]) else 0.0)
    exp_r, pf = np.array(exp_r), np.array(pf)
    e0 = strategy_metrics["expectancy_r"]
    return {
        "n_runs": n_runs, "strategy_expectancy_r": e0,
        "random_expectancy_r": {"mean": float(exp_r.mean()), "p5": float(np.percentile(exp_r, 5)),
                                "p50": float(np.percentile(exp_r, 50)), "p95": float(np.percentile(exp_r, 95))},
        "random_profit_factor_median": float(np.median(pf)),
        "p_value_expectancy": float((1 + (exp_r >= e0).sum()) / (n_runs + 1)),
    }


def ema_cross_benchmark(market: MarketData, cfg: RunConfig, window: Window = (None, None),
                        fast: int = 20, slow: int = 50) -> dict:
    """Naive benchmark: H1 EMA(fast/slow) cross, same filters/exits/risk as the real system."""
    ov = {}
    for s in market.symbols:
        ind = market.indicators(s, cfg.strategy)
        c = pd.Series(ind.close)
        ef, es = c.ewm(span=fast, adjust=False).mean(), c.ewm(span=slow, adjust=False).mean()
        up = ((ef > es) & (ef.shift(1) <= es.shift(1))).to_numpy()
        dn = ((ef < es) & (ef.shift(1) >= es.shift(1))).to_numpy()
        mask = entry_filter_mask(ind, cfg.strategy, market.specs[s])
        ov[s] = (up & mask, dn & mask)
    _, m = run_cfg(market, cfg, window, signal_override=ov)
    return m


def buy_and_hold(market: MarketData, window: Window = (None, None)) -> dict:
    """Unlevered equal-weight long basket of the traded symbols (price return only, no carry)."""
    lo, hi = market.slice_ns(*window)
    curves, per = [], {}
    for s in market.symbols:
        m = (market.t[s] >= lo) & (market.t[s] < hi)
        c = pd.Series(market.c[s][m], index=pd.to_datetime(market.t[s][m], unit="ns", utc=True))
        if len(c) < 2:
            continue
        per[s] = float(c.iloc[-1] / c.iloc[0] - 1) * 100
        curves.append(c / c.iloc[0])
    if not curves:
        return {}
    basket = pd.concat(curves, axis=1).ffill().bfill().mean(axis=1)
    dd_money, dd_pct = drawdown_stats(basket)
    return {"per_symbol_return_pct": per, "basket_return_pct": float((basket.iloc[-1] - 1) * 100), "basket_max_dd_pct": dd_pct,
            "note": "price-only, ignores carry; FX buy&hold is not a meaningful directional benchmark"}


# ------------------------------------------------------------------ regime / concentration analysis
def _grp(g: pd.DataFrame) -> pd.Series:
    w, l = g.pnl[g.pnl > 0].sum(), -g.pnl[g.pnl <= 0].sum()
    return pd.Series({"n": len(g), "expectancy_r": g.r.mean(), "profit_factor": (w / l) if l > 0 else float("inf"),
                      "win_rate": (g.pnl > 0).mean(), "net_pnl": g.pnl.sum()})


def regime_breakdown(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if trades.empty:
        return {}
    t = trades.copy()
    t["year"] = t.exit_time.dt.year
    t["vol_regime"] = t.regime.map({0: "low_vol", 1: "mid_vol", 2: "high_vol", -1: "unknown"})
    t["session"] = np.where(t.entry_time.dt.hour < 12, "london_am", "ny_overlap_pm")
    return {k: t.groupby(k)[["pnl", "r"]].apply(lambda g: _grp(g)).round(3)
            for k in ("year", "vol_regime", "side", "symbol", "session")}


def concentration_flags(trades: pd.DataFrame) -> dict:
    """Is the profit dependent on a handful of trades / one year?"""
    if len(trades) < 20 or trades.pnl.sum() <= 0:
        return {"applicable": False}
    pnl = trades.pnl.sort_values(ascending=False).to_numpy()
    top = max(1, int(math.ceil(0.05 * len(pnl))))
    by_year = trades.groupby(trades.exit_time.dt.year).pnl.sum()
    total = trades.pnl.sum()
    return {
        "applicable": True,
        "top5pct_trades_share_of_net": float(pnl[:top].sum() / total),
        "best_year_share_of_net": float(by_year.max() / total),
        "profitable_years": int((by_year > 0).sum()), "n_years": int(len(by_year)),
        "flag_few_trades_dominate": bool(pnl[:top].sum() / total > 1.5),
        "flag_single_year_dominates": bool(by_year.max() / total > 0.6 and len(by_year) >= 3),
    }
