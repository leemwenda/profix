# 4. Configuration parameters

Single source of truth: `config/strategy.toml`. Every field there has a matching, identically-
named MQL5 `input` (ProFX.mq5) and Pine `input.*` (ProFX_Trend_Breakout.pine). Python rejects
unknown keys on load (`ConfigError`); dataclasses validate ranges in `__post_init__` (e.g.
`risk_pct` is hard-capped at 2% regardless of what a config file says).

| Group | Key | Default | Meaning |
|---|---|---|---|
| strategy | donchian_n | 24 | breakout lookback, bars |
| strategy | atr_period | 14 | ATR smoothing period |
| strategy | ema_fast/slow | 50/200 | HTF trend EMAs |
| strategy | sl_atr_mult | 1.5 | stop distance in ATR |
| strategy | tp_r | 3.0 | take-profit in R (0=off) |
| strategy | be_r / trail_start_r | 1.0 / 1.5 | management triggers in R (0=off) |
| risk | risk_pct | 0.5 | % equity risked per trade (hard cap 2%) |
| risk | max_drawdown_pct | 8.0 | equity-peak breaker |
| risk | daily_loss_limit_pct | 2.0 | daily breaker |
| costs | spread/slippage/commission/swap | see file | execution realism knobs, also used for stress tests |

Full parameter list, types and validation rules: read `python/profx/config.py` — it is the
executable specification and stays more accurate than any table duplicated here.

**Never** hardcode account numbers, broker passwords, or the webhook secret in these files. The
webhook reads `WEBHOOK_SECRET` etc. from the environment only (see `06_security_checklist.md`).
