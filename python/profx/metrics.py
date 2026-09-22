"""Performance statistics. Never judge a system by net profit alone."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .engine import BacktestResult


def max_consecutive(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def drawdown_stats(equity: pd.Series) -> tuple[float, float]:
    """(max drawdown in money, max drawdown in % of running peak) from an equity series."""
    if len(equity) == 0:
        return 0.0, 0.0
    peak = equity.cummax()
    dd = peak - equity
    return float(dd.max()), float((dd / peak).max() * 100.0)


def tstat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 3 or np.std(x, ddof=1) == 0:
        return 0.0
    return float(np.mean(x) / (np.std(x, ddof=1) / math.sqrt(len(x))))


def compute_metrics(res: BacktestResult) -> dict:
    tr, eq = res.trades, res.equity
    out: dict = {"initial_balance": res.initial_balance, "halted": res.halted}
    n = len(tr)
    out["total_trades"] = n
    dd_money, dd_pct = drawdown_stats(eq)
    out["max_drawdown_money"], out["max_drawdown_pct"] = dd_money, dd_pct
    final = float(eq.iloc[-1]) if len(eq) else res.initial_balance
    out["final_equity"] = final
    out["net_profit"] = final - res.initial_balance
    out["return_pct"] = out["net_profit"] / res.initial_balance * 100.0
    out["exposure"] = res.exposure
    if n == 0:
        out.update(dict(gross_profit=0.0, gross_loss=0.0, profit_factor=float("nan"), expectancy_money=0.0,
                        expectancy_r=0.0, win_rate=float("nan"), avg_win=0.0, avg_loss=0.0, tstat_r=0.0,
                        max_consecutive_losses=0, sharpe=float("nan"), sortino=float("nan"), calmar=float("nan"),
                        recovery_factor=float("nan"), avg_holding_hours=0.0, worst_trade_r=0.0, cagr_pct=float("nan")))
        return out
    pnl = tr["pnl"].to_numpy(float)
    r = tr["r"].to_numpy(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gp, gl = float(wins.sum()), float(-losses.sum())
    out.update(
        gross_profit=gp, gross_loss=gl, profit_factor=(gp / gl) if gl > 0 else float("inf"),
        expectancy_money=float(pnl.mean()), expectancy_r=float(r.mean()), tstat_r=tstat(r),
        win_rate=float((pnl > 0).mean()), avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        max_consecutive_losses=max_consecutive(pnl <= 0), max_consecutive_wins=max_consecutive(pnl > 0),
        worst_trade_r=float(r.min()), best_trade_r=float(r.max()),
        avg_holding_hours=float(tr["bars_held"].mean()),
        recovery_factor=(out["net_profit"] / dd_money) if dd_money > 0 else float("nan"),
        total_commission=float(tr["commission"].sum()), total_swap=float(tr["swap"].sum()),
    )
    out["payoff_ratio"] = abs(out["avg_win"] / out["avg_loss"]) if out["avg_loss"] else float("nan")

    daily = eq.resample("1D").last().dropna()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    out["years"] = yrs
    out["cagr_pct"] = ((final / res.initial_balance) ** (1 / yrs) - 1) * 100.0 if final > 0 else -100.0
    out["calmar"] = out["cagr_pct"] / dd_pct if dd_pct > 0 else float("nan")
    ret = daily.pct_change().dropna()
    enough = len(ret) >= 60 and n >= 30
    out["ratios_reliable"] = bool(enough)
    if len(ret) > 2 and ret.std() > 0:
        out["sharpe"] = float(ret.mean() / ret.std() * math.sqrt(252))
        dn = ret[ret < 0]
        out["sortino"] = float(ret.mean() / math.sqrt((dn**2).mean()) * math.sqrt(252)) if len(dn) else float("nan")
    else:
        out["sharpe"] = out["sortino"] = float("nan")
    return out


def monthly_returns(equity: pd.Series, initial: float) -> pd.DataFrame:
    """Year x month table of % returns from the equity curve."""
    if len(equity) == 0:
        return pd.DataFrame()
    m = equity.resample("ME").last()
    prev = m.shift(1)
    prev.iloc[0] = initial
    pct = (m / prev - 1.0) * 100.0
    tab = pd.DataFrame({"year": pct.index.year, "month": pct.index.month, "ret": pct.values})
    return tab.pivot(index="year", columns="month", values="ret").round(2)


def format_metrics(m: dict) -> str:
    def f(k, fmt="{:,.2f}"):
        v = m.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        return fmt.format(v)

    rows = [
        ("Trades", f("total_trades", "{:d}")), ("Net profit", f("net_profit")), ("Return %", f("return_pct")),
        ("CAGR %", f("cagr_pct")), ("Gross profit", f("gross_profit")), ("Gross loss", f("gross_loss")),
        ("Profit factor", f("profit_factor")), ("Expectancy (money)", f("expectancy_money")),
        ("Expectancy (R)", f("expectancy_r", "{:.3f}")), ("t-stat of R", f("tstat_r")),
        ("Win rate", f("win_rate", "{:.1%}")), ("Avg win", f("avg_win")), ("Avg loss", f("avg_loss")),
        ("Max DD %", f("max_drawdown_pct")), ("Max DD money", f("max_drawdown_money")),
        ("Max consecutive losses", f("max_consecutive_losses", "{:d}")), ("Sharpe (daily, ann.)", f("sharpe")),
        ("Sortino", f("sortino")), ("Calmar", f("calmar")), ("Recovery factor", f("recovery_factor")),
        ("Avg holding (bars/h)", f("avg_holding_hours", "{:.1f}")), ("Exposure (time in market)", f("exposure", "{:.1%}")),
        ("Halted", str(m.get("halted"))),
    ]
    w = max(len(a) for a, _ in rows)
    return "\n".join(f"{a:<{w}}  {b}" for a, b in rows)
