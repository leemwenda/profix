# 10. Installation

## Python (research/backtest/webhook) - Windows/Linux/macOS
```
cd python
python3 -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q                                 # should show "68 passed"
python scripts/run_backtest.py --synthetic 4 --out out/demo --plot
python scripts/run_research.py --synthetic 6 --out out/research_demo   # pipeline demo, NOT real edge
```
`MetaTrader5` (for `export_mt5_history.py` and the live `MT5Executor`) is Windows-only and only
installs where `sys_platform == "win32"`; the rest of the stack runs anywhere.

## MT5 (Windows)
1. Copy `mt5/MQL5/Include/ProFX/` to `<Terminal Data Folder>/MQL5/Include/ProFX/`.
2. Copy `mt5/MQL5/Experts/ProFX/ProFX.mq5` to `<Terminal Data Folder>/MQL5/Experts/ProFX/`.
   (File -> Open Data Folder in MT5 to find the path.)
3. Open `ProFX.mq5` in MetaEditor, **Compile** (F7). Fix any errors — this build has not been
   compiled in this environment; treat the first compile as mandatory, not optional.
4. In MT5: Tools -> Options -> Expert Advisors -> "Allow algorithmic trading". Drag `ProFX` onto a
   chart (symbol/timeframe don't matter much — the EA trades its own `InpSymbols` list).
5. Defaults leave `InpMode=DRYRUN`. Verify the "ProFX: DRY RUN" comment shows on the chart and
   `MQL5/Files/ProFX/logs/*.jsonl` (or Common Files if `InpUseCommonFiles=true`) fills with
   `dry_run_order` events before ever flipping to TRADE mode.

## TradingView (Pine v6)
1. Open the Pine Editor, paste `pine/ProFX_Trend_Breakout.pine`, Add to Chart.
2. Use the Strategy Tester tab to review backtest stats (informational only — see limitations).
3. Set an alert on the strategy with condition "Any alert() function call", webhook URL pointing
   at your running webhook's `/webhook` endpoint, and fill in the `webhookSecretHint` input with
   your real secret **only in your private chart layout** — never share/publish a script with the
   secret filled in.

## Webhook
```
cd python
export WEBHOOK_SECRET=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
export WEBHOOK_SYMBOLS="EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD"
python scripts/run_webhook.py --host 127.0.0.1 --port 8000    # dry-run by default
curl -s localhost:8000/health
```
Put a TLS-terminating reverse proxy (nginx/Caddy) or a tunnel (Tailscale/ngrok with auth) in front
of it; never expose it directly. See `06_security_checklist.md` before setting `WEBHOOK_LIVE=1`.

## Backtesting / optimization / walk-forward
`python scripts/run_research.py --data-dir data --server-tz <broker tz> --out out/research` runs
the entire methodology (split, grid search, validation, sensitivity, stress, controls,
walk-forward, Monte Carlo, one-shot OOS, verdict). See `08_demo_testing_procedure.md` for the full
backtest -> demo -> live sequence.
