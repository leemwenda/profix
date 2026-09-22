# 8. Demo / forward-testing procedure (backtest -> demo -> live checklist)

1. **Get real data.** `python scripts/export_mt5_history.py --symbols EURUSD GBPUSD USDJPY AUDUSD USDCAD --years 6 --out data` (Windows, MT5 terminal running, `pip install MetaTrader5`). This also
   exports real symbol specs and swaps — update `config/strategy.toml` costs from them.
2. **Run research once**, honestly: `python scripts/run_research.py --data-dir data --server-tz <your broker's tz> --out out/research`. Read `out/research/research_report.md`. If the verdict is
   `REJECT`, stop and revisit the hypothesis — do not re-run OOS (the guard blocks it; `--force-oos`
   exists only for deliberate, documented exceptions and voids the OOS test's validity).
3. **Compile in MetaEditor.** Fix any compiler errors/warnings (this build was never compiled).
   Run the MT5 **Strategy Tester** (real-tick or every-tick model) over the same OOS window; sanity
   check trades roughly match `oos_trades.csv` in shape (exact match isn't expected — different
   engines — but net direction and rough trade count should be plausible).
4. **Attach the EA to a DEMO account**, `InpMode=DRYRUN` first for a few days: confirm the JSONL
   logs show sensible signals/rejections and no errors, with zero real orders.
5. **Switch to `InpMode=TRADE`, `InpConfirmLive=true` on the SAME demo account** for 4–8 weeks.
   Track: does live behavior match the backtest's exposure, trade frequency, and rough R
   distribution? Any decision to change the config resets the meaningfulness of "4-8 weeks."
6. **If using the webhook/TradingView path**: run the webhook with `WEBHOOK_LIVE` unset (dry-run)
   against demo Pine alerts first; confirm signals are validated and logged correctly end-to-end
   before setting `WEBHOOK_LIVE=1` against `MT5Executor` on the demo account.
7. **Only after demo results are consistent with backtest expectations**, and you have decided how
   much capital you can afford to lose, consider a small live allocation — start at a fraction
   (e.g. 10-25%) of your intended size, and keep `max_drawdown_pct`/`daily_loss_limit_pct` tight
   for the first live weeks.
8. **Re-run the whole research pipeline periodically** (e.g. quarterly) as new data accrues, with a
   *new* OOS window each time — do not keep testing against the same OOS segment.
