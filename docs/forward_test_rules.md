# MTF forward test rules (written before the test starts)

Strategy: H1 regime + M15 setup, M5 and M1 not required. Code: branch proFX-ai-v2, commit 713ff01.
Account: DEMO only.
Symbol: XAUUSD.
Start date: ____

## Fixed during the test (no changes until the test ends)
- Risk per trade: ____ % of equity
- Config: config/xauusd.toml, unchanged
- No new filters, no parameter changes, no skipped signals

## Costs to record for every trade
- Actual spread at entry, actual fill price vs signal price, exit slippage

## Pass / stop rules (decided now)
- Minimum sample: 100 trades
- Continue past 100 trades only if profit factor > 1.2
- Hard stop: if drawdown reaches ____ %, stop and review
- If spreads or fills are consistently worse than the backtest assumed (20 points spread floor, 2 and 5 pips slippage), note it and re-run the cost test with the real numbers

## What this test cannot show
- Anything about periods with different market conditions. A pass here is not proof of an edge.
