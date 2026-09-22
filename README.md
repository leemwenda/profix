# ProFX — Trend-Filtered Donchian Breakout, Multi-Platform Forex System

A deterministic, risk-managed trend-breakout strategy implemented three times over the same
mathematical rules: **MQL5** (MetaTrader 5 EA), **Pine Script v6** (TradingView), and **Python**
(research/backtest/webhook). Built for robustness and auditability, not for an impressive backtest
number. **No profitability is claimed or implied anywhere in this project.**

Start here:
- `docs/01_strategy_specification.md` — what the strategy does, in plain language
- `docs/02_mathematical_rules.md` — every rule as a formula
- `docs/07_known_limitations.md` — **read this before trusting anything else in the repo**
- `docs/10_installation.md` — set everything up
- `docs/08_demo_testing_procedure.md` — the mandatory path from backtest to real money

## Quick start (research pipeline, synthetic data — pipeline demo only)
```
cd python
pip install -r requirements.txt
python -m pytest tests -q                                        # 68 passed
python scripts/run_research.py --synthetic 6 --out out/demo
cat out/demo/research_report.md
```

## What's implemented
- **MQL5 EA** (`mt5/MQL5/`): multi-symbol, OnTick/OnTimer/OnTradeTransaction, magic-number scoped,
  structured JSONL logging, dry-run by default, GlobalVariable-persisted risk state.
- **Pine v6 strategy** (`pine/`): non-repainting signals, risk-based position sizing, JSON alerts
  for the webhook, documented limitations vs. the MQL5/Python engines.
- **Python engine** (`python/profx/`): realistic multi-symbol backtester (spread/slippage/
  commission/swap/execution-delay), 68 passing tests including look-ahead-bias mutation testing,
  a full robustness toolkit (Monte Carlo, walk-forward, parameter sensitivity, random-entry
  control, cost stress, benchmarks), and a one-shot-OOS research pipeline that outputs a
  REJECT / INCONCLUSIVE / CANDIDATE_FOR_DEMO_FORWARD_TEST verdict — never "profitable".
- **Webhook** (`python/profx/webhook/`): FastAPI service validating TradingView alerts (HMAC or
  shared-passphrase auth, schema + semantic checks, SQLite-backed replay protection, rate limiting,
  kill switch) before an executor (dry-run by default) touches MT5.

## Honest status (see `docs/09_final_code_review_checklist.md` for the full list)
- All Python code is exercised by 68 passing automated tests in this session.
- The MQL5 code has **not been compiled** (no MetaEditor available in this sandbox) — compiling it
  is the mandatory next step, not optional polish.
- The Pine script has **not been run in TradingView**.
- All current results are on **synthetic data** — there is no real-market evidence of edge yet.
  Get real broker history (`scripts/export_mt5_history.py`) and run the research pipeline again
  before drawing any conclusion.
# profix
