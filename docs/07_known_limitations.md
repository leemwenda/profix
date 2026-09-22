# 7. Known limitations (read before trusting this system with money)

**No amount of backtesting proves future profitability.** This document exists so limitations are
stated up front, not discovered after a loss.

1. **Synthetic data everywhere in this build.** No real broker history was available in this
   environment. Every number this repository can currently produce comes from
   `profx/data.py::synthetic_fx`, a regime-switching random-walk generator built only to exercise
   the pipeline. **Run `run_research.py` on your own broker's real H1 history (see
   `08_demo_testing_procedure.md`) before drawing any conclusion about edge.**
2. **HTF EMA parity between Python/MQL5/Pine is not empirically verified.** The three
   implementations follow the same formula (§2.3) but were not cross-checked bar-by-bar (no MT5
   terminal or Pine runtime is available here). Do this reconciliation yourself before relying on
   the EA or Pine script to match backtest behaviour.
3. **Currency-exposure cap is a coarse correlation proxy**, not a real covariance model (§2.7).
   Do not scale up `max_open_positions` or the symbol universe without replacing it.
4. **Swap rates are placeholder assumptions** (`swap_long/short_per_lot_night = -3.0`) — replace
   with your broker's real values (exported by `export_mt5_history.py`) before trusting overnight
   P&L in a backtest.
5. **Cost stress and Monte Carlo model parameter uncertainty, not model-form risk.** They cannot
   catch a wrong hypothesis, only fragility of a given one. A "CANDIDATE" verdict from
   `research.py` is a licence to demo-forward-test, not a profitability guarantee.
6. **The 15-minute-rounded UTC-offset heuristic** (`PFX_UtcOffsetSec` in MQL5, or the
   `server_tz`/`server_utc_offset_hours` choice in `data.load_csv`) can be wrong for brokers with
   unusual server-time conventions, especially in the week each side of a DST change. Verify your
   broker's actual H4 bar-open times against `htf_offset_hours`.
7. **Weekend/holiday gap handling is a generic day-of-week filter**, not a holiday calendar; major
   holidays with early closes are not modelled and can produce unrealistic backtest fills around
   them.
8. **Pine execution realism is inferior to the MQL5 backtest/engine.py**: TradingView's Strategy
   Tester cannot see real broker spread, exact tick value or margin; treat Pine as a signal/alert
   generator, not the risk-authoritative implementation.
9. **The MQL5 code has not been compiled** (no MetaEditor/Wine available in this environment). It
   has been written to compile and carefully reviewed, but treat the first compile + Strategy
   Tester run as part of the validation process, not a formality (see `07a` checklist item in
   `09_final_code_review_checklist.md`).
10. **A positive backtest, even out-of-sample, is not a green light for real money.** Forward-test
    on a demo account per `08_demo_testing_procedure.md` first, and start any live account at a
    fraction of the intended size.
