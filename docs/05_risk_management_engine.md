# 5. Risk-management engine

## Position sizing (never fixed lots)
`lots = floor(risk_pct%·equity / (stop_distance·contract_value + commission)  /  lot_step) · lot_step`
Implemented identically (modulo language) in `python/profx/engine.py::_try_enter`,
`RiskManager.mqh::CalcLots`, and Pine's `qtyLong`/`qtyShort`. Always rounds **down**; a lot size
below the broker minimum means the trade is **skipped**, never bumped up (`round_lots_down`,
`PFX_FloorLots`, tested in `test_lot_below_broker_minimum_skips_trade_instead_of_rounding_up` and
`test_round_lots_never_up`).

## Hard safety ceilings (cannot be configured away)
- `risk_pct` > 2% is rejected at config load time (`RiskParams.__post_init__`).
- `max_total_risk_pct` must be ≥ `risk_pct` (can't have a 1-trade risk exceed the portfolio cap).
- A trade is never opened without an initial stop loss (webhook: `sl_required`; EA: stop distance
  is always computed before sizing; engine: `D = sl_atr_mult*atr` is required for `lots>0`).

## Circuit breakers
1. **Daily loss limit** (2% default): blocks new entries for the rest of the UTC day; optionally
   flattens open positions (`flatten_on_daily_loss=true` by default).
2. **Max drawdown** (8% default, from peak equity): flattens everything and **halts** new entries.
   - Backtest/research: `dd_halt_cooldown_days` can simulate an operator resetting after N days —
     this is a research knob only (documented as such in `config.py`); **live default is 0
     (never auto-resets)**.
   - Live (MQL5): `CRiskManager::ResetHalt()` exists but is **not** called automatically anywhere
     in `ProFX.mq5`. A human must edit the EA (or a future admin control) to call it, after
     investigating why the breaker tripped.
3. **Spread gate**: rejects entries when the spread exceeds 3 pips OR 15% of the stop distance
   (whichever is tighter) — protects against thin-liquidity/news-spike spreads.
4. **Margin gate**: refuses a new trade if projected free margin after the trade would fall below
   50% of equity.
5. **Currency exposure cap**: refuses a new trade if it would push same-direction exposure to a
   base or quote currency above 2 open positions.
6. **Cooldown / daily trade cap**: 4-bar cooldown per symbol after any close; max 4 new trades/day
   across the whole portfolio.

## What "flatten" means at each breaker
Positions are closed at the **next bar open** (backtest) or **at market on the next `OnTick`**
(EA) — never mid-formula, never by silently widening a stop. `PFX_SetExitReason` /
`ps.pending_close` record *why* so the log/trade history is auditable.

## Explicit non-goals
This is not a portfolio VaR/covariance risk engine. The currency-exposure cap is a coarse
correlation proxy (§2.7 in the math doc) — for a materially larger symbol universe, replace it
with a real covariance-based exposure check before increasing `max_open_positions`.
