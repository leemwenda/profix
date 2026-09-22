# 3. System architecture

```
profx/
├── mt5/MQL5/
│   ├── Include/ProFX/   Defs, Utils, Logger, Indicators, RiskManager, TradeManager, Config  (.mqh)
│   └── Experts/ProFX/   ProFX.mq5                       <- the Expert Advisor
├── pine/                ProFX_Trend_Breakout.pine        <- TradingView v6 strategy/alerts
├── python/profx/
│   ├── config.py         typed, validated configuration (dataclasses; TOML loader)
│   ├── specs.py           symbol contract specs, lot rounding
│   ├── data.py             CSV loading, HTF resampling, validation, synthetic data generator
│   ├── signals.py           pure, vectorised, causal indicator + signal functions
│   ├── engine.py             event-driven multi-symbol backtest engine (fills, costs, risk, mgmt)
│   ├── metrics.py             performance statistics
│   ├── robustness.py           Monte Carlo, stress tests, sensitivity, controls, benchmarks
│   ├── research.py              split -> select -> confirm -> OOS-once pipeline, verdict
│   └── webhook/                  FastAPI signal-validation service (settings/store/validation/executor/app)
├── python/scripts/       run_backtest.py, run_research.py, export_mt5_history.py, run_webhook.py
├── python/tests/         68 tests: fills, no-look-ahead, risk/portfolio, webhook
├── config/strategy.toml  single source of truth for default parameters
└── docs/                 this documentation set
```

## Why this split
- **Signal generation is separated from execution** in every implementation: `signals.py` has no
  knowledge of orders, money or state; `engine.py` (or `TradeManager.mqh` / Pine's `strategy.*`
  calls) is where execution happens. This lets the strategy logic be tested in isolation
  (`test_no_lookahead.py`, `test_engine_fills.py` construct scenarios without touching real I/O).
- **Risk is a separate module from execution** (`RiskManager.mqh` / `robustness.py` sizing +
  `engine.py`'s gate functions): position sizing and the circuit breakers must be auditable
  independently of order-sending mechanics.
- **The webhook is a separate process from both MT5 and TradingView**: TradingView cannot call MT5
  directly, so Pine emits an alert (JSON) → a webhook validates it (auth, schema, replay, risk
  ceiling) → an executor (dry-run by default, or the real `MetaTrader5` package) places the order.
  This validation layer exists specifically so a compromised or spoofed TradingView alert cannot
  move money without independent checks.
- **Three implementations of the same rules, one source of truth**: `config/strategy.toml` values
  must match the MQL5 `input` defaults and the Pine `input.*` defaults. This is checked by
  `test_default_toml_loads_and_equals_dataclass_defaults`, but the MQL5/Pine copies are NOT
  automatically verified against Python (no compiler available in this environment) — reconcile
  them manually against this file whenever a default changes, and forward-test on demo before
  trusting parity.

## Data flow (live)
```
MT5 terminal (EA)                         TradingView (Pine)
   OnTick/OnTimer                             barstate.isconfirmed
      -> Indicators.mqh (closed bars only)        -> alert() JSON
      -> RiskManager.mqh (size + gates)               |
      -> TradeManager.mqh (CTrade, retries,            v
         duplicate guard, verify-after-send)      webhook/app.py
      -> Logger.mqh (JSONL)                          -> auth (HMAC or passphrase)
                                                       -> schema + semantic validation
                                                       -> replay/duplicate check (SQLite)
                                                       -> rate limit, kill switch
                                                       -> executor (DryRun default / MT5Executor)
```
