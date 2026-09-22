# 1. Strategy specification

## 1.1 Hypothesis (state this explicitly, so it can be proven wrong)
On liquid major FX pairs, price that closes beyond its recent N-bar range, in the direction
of a persistent higher-timeframe trend, with a strong closing position within the bar (not a
wick-driven spike), is more likely to continue than reverse over the next several hours, enough
to overcome spread + commission + slippage after a volatility-scaled stop and 3R target.

This is an ordinary, well-known breakout/trend-following hypothesis. It is **not** claimed to be
novel or reliably profitable — see `07_known_limitations.md` and the research verdict produced
by `run_research.py` before trusting it with money.

## 1.2 Instruments, sessions, timeframe
- Symbols: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD (configurable; add pairs with real spread/liquidity
  data - see `02_mathematical_rules.md` §2.7 for why untested pairs must not just be added blindly).
- Execution timeframe: H1. Decisions are made at the **close** of an H1 bar; orders fill at the
  **open** of the next H1 bar.
- Trend timeframe: H4, using only **closed** H4 bars.
- Session: entries only when the *next* (entry) bar opens between 07:00–19:00 UTC, and not after
  17:00 UTC on Fridays. No entries Saturday/Sunday.

## 1.3 Trend / structure definition (deterministic)
HTF trend = +1 (up) if EMA(fast=50) > EMA(slow=200) **and** EMA(fast) now > EMA(fast) `slopeBars`
(6) H4 bars ago. Trend = −1 (down) under the mirror condition. Otherwise 0 (no trade).

## 1.4 Entry condition (deterministic)
Long: `close > Donchian_High(24, shift 1) + 0.05*ATR(14)` **and** close-location-value
`(close−low)/(high−low) ≥ 0.60` **and** HTF trend == +1 **and** volatility/session filters pass.
Short is the mirror. Donchian levels use the 24 bars **before** the signal bar (shift 1), never
including the signal bar itself.

## 1.5 Invalidation
No signal (no trade) if: ATR(14) in pips is outside [5, 50]; the signal bar's range exceeds
2.5×ATR (exhaustion-bar guard); outside the session window; HTF trend is 0 or opposes the
breakout direction; CLV filter fails (a close near the wrong end of the bar).

## 1.6 Stops, targets, management
- Initial SL = 1.5 × ATR(14) from the fill price.
- TP = 3R (0 disables a fixed TP, i.e. trail-only).
- Break-even: once price is favourable by ≥ 1.0R, move SL to entry + 1 pip (never loosens).
- Trailing: once favourable by ≥ 1.5R, trail at `close − 2.0×ATR` (long) / mirror (short); only
  ever tightens.
- Partial close: disabled by default (`partial_pct=0`); if enabled, closes `partial_pct` of the
  position at `partial_r` R.
- Time exit: close if still open after 72 H1 bars.
- Trend-flip exit: close if the HTF trend flips against the position.
- Weekend: flatten by 20:00 UTC Friday.

## 1.7 Costs, limits, caps (see `02_mathematical_rules.md` for the exact formulas)
Max spread 3 pips (and ≤15% of the SL distance) · slippage and commission modelled explicitly ·
min/max lot and lot step from the real symbol · risk 0.5% of equity per trade, hard-capped at 2% ·
max 3 open positions total, 1 per symbol · max total open risk 1.5% of equity · max 4 trades/day ·
4-bar cooldown per symbol after a close · daily loss limit 2% (flattens and blocks new entries for
the rest of the day) · max drawdown 8% from peak equity (flattens everything and halts until a
human resets it) · max 2 same-direction exposures per currency (crude correlation control) ·
minimum 50% free margin after a new trade.

## 1.8 Explicitly deterministic — no vague terms
"Strong trend", "good setup" etc. are not used anywhere; every condition above is a formula over
OHLC, ATR and EMA values, computed the same way in Python, MQL5 and Pine (see
`02_mathematical_rules.md` for the shared formulas and the cross-implementation parity notes).
