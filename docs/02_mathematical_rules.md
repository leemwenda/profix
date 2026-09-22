# 2. Mathematical trading rules

Notation: subscript `i` = bar index, `shift 1` = the most recently **closed** bar. `pip` = 10×point
for 3/5-digit quotes, else 1×point.

## 2.1 True Range / ATR
`TR_i = max(H_i − L_i, |H_i − C_{i−1}|, |L_i − C_{i−1}|)`; `ATR = SMA(TR, 14)` (matches MT5 `iATR`).

## 2.2 Donchian levels (causal)
`HH_i = max(H_{i-1..i-N})`, `LL_i = min(L_{i-1..i-N})`, N = 24. **Excludes** bar `i` itself.

## 2.3 HTF trend
On H4 bars: `EF = EMA(close, 50)`, `ES = EMA(close, 200)`.
`trend = +1` if `EF > ES and EF_t > EF_{t-6}`; `−1` if the mirror; else `0`.
Only a **closed** H4 bar may be used: for exec-bar `i`, the last usable H4 bar is the most recent
one whose close time ≤ `open_time(i) + 1h`. See `signals.py::compute_indicators` for the exact
`searchsorted` implementation and `tests/test_no_lookahead.py` for the proof this holds even when
future bars are corrupted.

## 2.4 Entry
```
CLV_i = (C_i − L_i) / (H_i − L_i)                (0.5 if H_i == L_i)
long_i  = (C_i > HH_i + 0.05·ATR_i) and (CLV_i ≥ 0.60) and (trend_i == +1) and filters_i
short_i = (C_i < LL_i − 0.05·ATR_i) and (CLV_i ≤ 0.40) and (trend_i == −1) and filters_i
filters_i = (5 ≤ ATR_i/pip ≤ 50) and session(i+1) and (H_i − L_i ≤ 2.5·ATR_i)
```
Fill: bar `i+1` open, `+ spread` (long) `/ 0` adjustment (short is quoted at bid).

## 2.5 Stops / targets
```
D        = 1.5 · ATR_i                     (distance, price units)
SL       = fill − side·D
TP       = fill + side·3·D                 (0 -> disabled)
BE (once fav ≥ 1.0·D):    SL := entry + side·1 pip     (only if it TIGHTENS)
Trail (once fav ≥ 1.5·D): SL := extreme − side·2.0·ATR  (only if it TIGHTENS)
```
`fav` = signed favourable excursion from entry. The **stop-never-loosens invariant** is enforced
in all three implementations and unit-tested (`test_trailing_stop_locks_profit_and_never_loosens`,
MQL5 `CTradeManager::ModifyStop`, Pine's tighten-only `if` guards).

## 2.6 Position sizing
```
lots_raw = risk_pct/100 · equity / (D · contract_size · quote_to_account_rate + commission_per_lot)
lots     = floor(lots_raw / lot_step) · lot_step        (NEVER rounds up; floor(x) not round(x))
```
If `lots < broker_min`, the trade is **skipped**, not upsized. `risk_money = lots·(D·value + comm)`
must always be ≤ `risk_pct% · equity` (unit-tested: `test_risk_never_exceeds_budget`).

## 2.7 Portfolio / exposure caps
```
open_positions          ≤ max_open_positions (3)
positions_per_symbol    ≤ 1
Σ open_risk_money + new_risk_money ≤ max_total_risk_pct% · equity   (1.5%)
same_direction_exposure(currency) + 1 ≤ max_same_ccy_dir (2)     [per base AND quote currency]
trades_today             ≤ max_daily_trades (4)
time_since_last_close(symbol) ≥ cooldown_bars · 1h (4h)
```
The currency-exposure cap is a **crude proxy** for correlation risk (it counts open positions
touching a currency in a given net direction); it is not a covariance-based portfolio VaR model.
Extending to more pairs without checking realised pairwise correlation risks understating true
combined exposure — see `07_known_limitations.md`.

## 2.8 Circuit breakers
```
drawdown_pct = (peak_equity − equity) / peak_equity · 100
if drawdown_pct ≥ max_drawdown_pct (8%):  flatten all, HALT new entries until a human resets it
if (equity/day_start_equity − 1)·100 ≤ −daily_loss_limit_pct (2%): flatten all (if configured),
   block new entries until the next UTC day
```
In the MQL5 EA, `max_drawdown_pct` halt is **latched**: `CRiskManager::ResetHalt()` must be called
deliberately (not automatically) — see `05_risk_management_engine.md`.
