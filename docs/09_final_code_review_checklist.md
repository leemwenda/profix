# 9. Final code-review checklist

## Logic / correctness
- [x] Look-ahead / repainting: 11 dedicated tests (`test_no_lookahead.py`), plus 2 independent
      injected-bug mutation checks confirmed to fail without the fix (done during this build).
- [x] Position sizing never exceeds the risk budget; never rounds lots up (`test_risk_*`,
      `test_round_lots_never_up`).
- [x] Stops never loosen (`stop_loosen_attempts` invariant, asserted in 3 tests + engine returns a
      violation flag rather than silently allowing it).
- [x] SL/TP-same-bar ambiguity resolved conservatively (stop wins) — `test_sl_and_tp_same_bar_*`.
- [x] Gaps through stops fill at the gap price + slippage, not the stale stop price —
      `test_gap_through_stop_*`.
- [x] Long stops checked against bid, short stops against ask (spread asymmetry) —
      `test_long_stop_checked_on_bid_short_stop_on_ask`.
- [x] Portfolio limits (max positions, per-symbol, currency exposure, cooldown, daily cap) unit
      tested against the full trade history of a multi-year, multi-symbol run.
- [x] Circuit breakers halt/block and are latched, not self-resetting, in both backtest and the
      MQL5 EA (dd_halt_cooldown_days documented as research-only).
- [x] Config validation rejects unsafe values (risk_pct>2%, inverted EMA periods, unknown TOML
      keys, etc.) — `test_config_rejects_unsafe_values`, `test_toml_unknown_keys_rejected`.
- [ ] **Floating-point / broker-specific constraints**: reviewed in Python (lot rounding via
      floor+epsilon), and mirrored in MQL5 (`PFX_FloorLots`, `PFX_NormPrice` uses tick_size
      rounding) — but not verified against a live broker's exact tick/lot-step edge cases. Do this
      during demo testing.
- [ ] **Time-zone / DST**: `PFX_UtcOffsetSec` rounds the server-vs-GMT offset to 15 minutes each
      tick; this is robust to normal DST jumps but **not verified against your specific broker**.
      Confirm your broker's H4 bar alignment against `htf_offset_hours` before trusting parity
      with the Python backtest.
- [ ] **Recovery after restart**: MQL5 risk state (peak equity, daily counters, halt flag,
      cooldown timestamps) is persisted via `GlobalVariable*` (survives EA reload, not a full
      terminal reinstall) — exercised in code but not tested live end-to-end (needs a running
      terminal).

## Process
- [x] OOS window can be evaluated exactly once per research run (`OOSGuard`); a second run raises
      `OOSAlreadyUsed` — verified (`2nd OOS blocked: OOSAlreadyUsed` in this session's smoke test).
- [x] Objective for parameter selection is t-stat of R with hard PF/drawdown/trade-count
      constraints, not net profit (`robust_score`).
- [x] Random-entry control, EMA-cross benchmark, buy-and-hold, walk-forward, Monte Carlo, cost
      stress and parameter-neighbour sensitivity are all wired into one pipeline
      (`research.run_research`) and produce a single non-hedging verdict (REJECT / INCONCLUSIVE /
      CANDIDATE_FOR_DEMO_FORWARD_TEST — never "profitable").
- [ ] **MQL5 has not been compiled** (no MetaEditor/Wine in this environment) — brace/paren
      balance was sanity-checked programmatically, and the code was written and re-read carefully,
      but a real compile in MetaEditor is a mandatory next step before any testing.
- [ ] **Pine script has not been run in TradingView** — same caveat; review the "KNOWN
      LIMITATIONS" block inside the script before use.
- [ ] **Cross-implementation parity (Python vs MQL5 vs Pine) is a design intent, not a proven
      fact.** Reconcile signal-by-signal on a demo run before trusting the EA/Pine to reproduce
      backtest behavior (see `07_known_limitations.md`, item 2).

## Test suite status (this session)
68/68 automated tests passing: 15 fill/cost/stop tests, 11 no-look-ahead tests (+2 confirmed-fatal
mutation checks), 15 risk/portfolio/config/metrics tests, 17 webhook tests, plus full end-to-end
research pipeline smoke tests on synthetic data. Run `python -m pytest python/tests -q` yourself.
