//+------------------------------------------------------------------+
//|                                                        ProFX.mq5   |
//|         Trend-filtered Donchian breakout, multi-symbol, MT5        |
//|  Mirrors python/profx (engine.py / signals.py / config.py) 1:1.    |
//|  DEFAULT MODE = DRY RUN. Read docs/installation.md before trading. |
//+------------------------------------------------------------------+
#property copyright "ProFX"
#property version   "1.00"
#property strict

#include <ProFX/Defs.mqh>
#include <ProFX/Utils.mqh>
#include <ProFX/Logger.mqh>
#include <ProFX/Indicators.mqh>
#include <ProFX/RiskManager.mqh>
#include <ProFX/TradeManager.mqh>
#include <ProFX/Config.mqh>

//======================================================== inputs ====
input string             InpSymbols            = "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD"; // Traded symbols (CSV, "" = chart symbol only)
input ENUM_PFX_MODE      InpMode               = PFX_MODE_DRYRUN;    // Mode
input bool               InpConfirmLive        = false;              // I understand this places REAL orders (required for TRADE mode)
input long               InpMagic              = 26092101;           // Magic number
input ENUM_TIMEFRAMES    InpTfExec             = PERIOD_H1;          // Execution timeframe
input ENUM_TIMEFRAMES    InpTfTrend            = PERIOD_H4;          // HTF trend timeframe

input group "=== Signal ==="
input int    InpDonchianN            = 24;
input int    InpAtrPeriod            = 14;
input int    InpEmaFast              = 50;
input int    InpEmaSlow              = 200;
input int    InpSlopeBars            = 6;
input double InpBreakoutBufferAtr    = 0.05;
input double InpMinClv               = 0.60;
input double InpMaxBarRangeAtr       = 2.5;
input double InpMinAtrPips           = 5.0;
input double InpMaxAtrPips           = 50.0;
input int    InpSessionStartUtc      = 7;
input int    InpSessionEndUtc        = 19;
input int    InpFridayLastEntryUtc   = 17;
input bool   InpAllowLong            = true;
input bool   InpAllowShort           = true;

input group "=== Stops / exits ==="
input double InpSlAtrMult            = 1.5;
input double InpTpR                  = 3.0;   // 0 = no fixed TP
input double InpBeR                  = 1.0;   // 0 = disabled
input double InpBeOffsetPips         = 1.0;
input double InpTrailStartR          = 1.5;   // 0 = disabled
input double InpTrailAtrMult         = 2.0;
input double InpPartialR             = 1.5;
input double InpPartialPct           = 0.0;   // 0 = disabled, else 0..0.9
input int    InpMaxBarsInTrade       = 72;
input bool   InpExitOnTrendFlip      = true;
input bool   InpCloseBeforeWeekend   = true;
input int    InpWeekendCloseHourUtc  = 20;
input int    InpMaxEntryDelaySec     = 90;    // skip a signal if we could not act within N sec of new bar

input group "=== Risk ==="
input double InpRiskPct              = 0.5;   // % of equity/balance per trade, hard-capped at 2.0
input bool   InpRiskOnEquity         = true;
input int    InpMaxOpenPositions     = 3;
input double InpMaxTotalRiskPct      = 1.5;
input int    InpMaxDailyTrades       = 4;
input int    InpCooldownBars         = 4;
input double InpDailyLossLimitPct    = 2.0;
input double InpMaxDrawdownPct       = 8.0;
input int    InpMaxSameCcyDir        = 2;
input double InpMaxSpreadPips        = 3.0;
input double InpMaxSpreadFracOfSl    = 0.15;
input double InpMinFreeMarginPct     = 50.0;
input double InpCommissionPerLotRt   = 7.0;
input bool   InpFlattenOnDailyLoss   = true;

input group "=== Execution ==="
input int    InpDeviationPoints      = 20;
input int    InpMaxRetries           = 3;
input int    InpMinSecondsBetweenOrders = 5;
input int    InpMaxOrdersPerHour     = 20;
input int    InpMaxConsecutiveRejects = 5;
input int    InpMaxTickAgeSec        = 30;

input group "=== Logging ==="
input bool   InpVerboseLog           = false;
input bool   InpUseCommonFiles       = false;

//======================================================== state =====
string           g_symbols[];
CIndicators      g_ind[];
datetime         g_lastBarTime[];      // last exec-bar open time we ACTED on, per symbol
datetime         g_lastExitTime[];     // for cooldown
CLogger          g_log;
CRiskManager     g_risk;
CTradeManager    g_tm;
SStrategyParams  g_sp;
SRiskParams      g_rp;
SExecParams      g_ep;

//+------------------------------------------------------------------+
int OnInit()
{
   string err;
   g_sp = PFX_BuildStrategyParams(InpTfExec, InpTfTrend, InpDonchianN, InpAtrPeriod, InpEmaFast, InpEmaSlow,
      InpSlopeBars, InpBreakoutBufferAtr, InpMinClv, InpMaxBarRangeAtr, InpMinAtrPips, InpMaxAtrPips,
      InpSessionStartUtc, InpSessionEndUtc, InpFridayLastEntryUtc, InpAllowLong, InpAllowShort,
      InpSlAtrMult, InpTpR, InpBeR, InpBeOffsetPips, InpTrailStartR, InpTrailAtrMult, InpPartialR, InpPartialPct,
      InpMaxBarsInTrade, InpExitOnTrendFlip, InpCloseBeforeWeekend, InpWeekendCloseHourUtc, InpMaxEntryDelaySec);
   g_rp = PFX_BuildRiskParams(InpRiskPct, InpRiskOnEquity, InpMaxOpenPositions, InpMaxTotalRiskPct,
      InpMaxDailyTrades, InpCooldownBars, InpDailyLossLimitPct, InpMaxDrawdownPct, InpMaxSameCcyDir,
      InpMaxSpreadPips, InpMaxSpreadFracOfSl, InpMinFreeMarginPct, InpCommissionPerLotRt, InpFlattenOnDailyLoss);
   g_ep.magic = InpMagic; g_ep.deviationPoints = InpDeviationPoints; g_ep.maxRetries = InpMaxRetries;
   g_ep.minSecondsBetweenOrders = InpMinSecondsBetweenOrders; g_ep.maxOrdersPerHour = InpMaxOrdersPerHour;
   g_ep.maxConsecutiveRejects = InpMaxConsecutiveRejects; g_ep.maxTickAgeSec = InpMaxTickAgeSec;
   g_ep.commentPrefix = "PFX_";
   g_ep.dryRun = (InpMode == PFX_MODE_DRYRUN) || !InpConfirmLive;

   if(!PFX_ValidateConfig(g_sp, g_rp, err))
   {
      PrintFormat("ProFX: INVALID CONFIG - %s. EA will not run.", err);
      return INIT_PARAMETERS_INCORRECT;
   }

   g_log.Init(InpMagic, InpVerboseLog, InpUseCommonFiles, EnumToString(InpTfExec));
   g_risk.Init(g_rp, InpMagic, &g_log);
   g_tm.Init(g_ep, &g_log);

   if(InpMode == PFX_MODE_TRADE && !InpConfirmLive)
      g_log.Log("WARN", "startup", "*", "forced_dry_run",
                PFX_KVs("reason", "InpMode=TRADE but InpConfirmLive=false"));
   if(g_ep.dryRun)
      Comment("ProFX: DRY RUN (no real orders). Set InpMode=TRADE and InpConfirmLive=true to go live.");

   if(!BuildSymbolList()) return INIT_FAILED;

   ArrayResize(g_ind, ArraySize(g_symbols));
   ArrayResize(g_lastBarTime, ArraySize(g_symbols));
   ArrayResize(g_lastExitTime, ArraySize(g_symbols));
   for(int i = 0; i < ArraySize(g_symbols); i++)
   {
      if(!SymbolSelect(g_symbols[i], true) || !g_ind[i].Init(g_symbols[i], g_sp))
      {
         PrintFormat("ProFX: failed to initialise %s", g_symbols[i]);
         return INIT_FAILED;
      }
      g_lastBarTime[i] = 0;
      g_lastExitTime[i] = (datetime)PFX_GvGet(PFX_Key(InpMagic, "last_exit", g_symbols[i]), 0);
   }
   g_log.Log("INFO", "startup", "*", "init_ok",
            PFX_Join(PFX_KVs("version", PFX_VERSION), PFX_KVb("dry_run", g_ep.dryRun)));
   EventSetTimer(30);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   Comment("");
   g_log.Log("INFO", "shutdown", "*", "deinit", PFX_KVi("reason", reason));
}

bool BuildSymbolList()
{
   string src = (InpSymbols == "") ? _Symbol : InpSymbols;
   string parts[];
   int n = StringSplit(src, ',', parts);
   ArrayResize(g_symbols, 0);
   for(int i = 0; i < n; i++)
   {
      string s = parts[i];
      StringTrimLeft(s); StringTrimRight(s);
      if(s == "") continue;
      if(SymbolInfoInteger(s, SYMBOL_SELECT) < 0 && !SymbolSelect(s, true))
      {
         PrintFormat("ProFX: symbol %s not available at this broker", s);
         continue;
      }
      int m = ArraySize(g_symbols);
      ArrayResize(g_symbols, m + 1);
      g_symbols[m] = s;
   }
   if(ArraySize(g_symbols) == 0) { Print("ProFX: no valid symbols"); return false; }
   return true;
}

//======================================================== main loop =
void OnTick() { ProcessAll(); }
void OnTimer() { ProcessAll(); }   // safety net if ticks stall on a quiet symbol

void ProcessAll()
{
   g_risk.OnNewBar();   // cheap; internally only acts once per new UTC day
   string flattenReason;
   bool mustFlatten = g_risk.MustFlattenAll(flattenReason);

   for(int i = 0; i < ArraySize(g_symbols); i++)
   {
      string sym = g_symbols[i];
      ManageOpenPosition(sym, i, mustFlatten, flattenReason);
      if(!IsNewExecBar(sym, i)) continue;
      if(!mustFlatten) CheckTrendFlipOnNewBar(sym, i);   // once per new bar, not every tick
      if(mustFlatten) continue;                          // do not open anything while a breaker is tripped
      TryEnter(sym, i);
   }

   if(g_tm.TooManyConsecutiveRejects())
      g_log.Log("ERROR", "safety", "*", "too_many_rejects",
               PFX_KVs("action", "pausing new entries until terminal/EA is checked by the operator"));
}

bool IsNewExecBar(const string sym, const int i)
{
   datetime t0 = iTime(sym, g_sp.tfExec, 0);
   if(t0 == 0 || t0 == g_lastBarTime[i]) return false;
   // shift 1 = the bar that just closed; base every decision on ITS close, never on shift 0
   g_lastBarTime[i] = t0;
   return true;
}

//======================================================== entries ===
void TryEnter(const string sym, const int idx)
{
   if(g_tm.TooManyConsecutiveRejects()) return;
   if(g_tm.HasOpenPosition(sym)) return;

   // "we must act within N seconds of the new bar" - guards against a frozen/slow terminal
   // silently trading a stale signal far into the next bar
   datetime barOpen = iTime(sym, g_sp.tfExec, 1);
   if(g_sp.maxEntryDelaySec > 0 && (long)TimeTradeServer() - (long)(barOpen + PeriodSeconds(g_sp.tfExec)) > g_sp.maxEntryDelaySec)
   {
      g_log.Log("DEBUG", "signal", sym, "skip", PFX_KVs("reason", "entry_delay_exceeded"));
      return;
   }

   SSignal sig;
   if(!ComputeSignal(sym, idx, sig)) return;
   if(!sig.valid) return;

   SSpec spec;
   if(!PFX_LoadSpec(sym, spec)) { g_log.Log("ERROR", "signal", sym, "skip", PFX_KVs("reason", "spec_unavailable")); return; }

   MqlTick tick;
   if(!SymbolInfoTick(sym, tick)) return;
   double spreadPrice = tick.ask - tick.bid;
   double D = g_sp.slAtrMult * sig.atr;

   double fillGuess = (sig.dir > 0) ? tick.ask : tick.bid;   // used only for gate/size estimation
   double sl = fillGuess - sig.dir * D;
   string reason = g_risk.CheckEntryGates(sym, g_tm.CountOpenPositions(), g_tm.HasOpenPosition(sym) ? 1 : 0,
                    g_lastExitTime[idx], spreadPrice, D, CurrencyDirCount(sym, sig.dir, true),
                    CurrencyDirCount(sym, sig.dir, false));
   if(reason != "")
   {
      g_log.Log("INFO", "signal", sym, "rejected", PFX_Join(PFX_KVs("reason", reason), PFX_KVs("signal_id", sig.id)));
      return;
   }

   double lots = g_risk.CalcLots(spec, fillGuess, sl);
   if(lots < spec.volMin - 1e-9)
   {
      g_log.Log("INFO", "signal", sym, "rejected", PFX_KVs("reason", "lot_below_min"));
      return;
   }
   double riskMoney = g_risk.RiskMoney(spec, lots, fillGuess, sl);
   if(!g_risk.TotalRiskOk(g_tm.SumOpenRiskMoney(g_risk, sym), riskMoney))
   {
      g_log.Log("INFO", "signal", sym, "rejected", PFX_KVs("reason", "max_total_risk"));
      return;
   }
   // OrderCalcMargin is the broker-authoritative margin calculation (handles cross pairs, leverage tiers, etc.)
   double marginCheck = 0.0;
   long leverage = AccountInfoInteger(ACCOUNT_LEVERAGE);
   if(!OrderCalcMargin(sig.dir > 0 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL, sym, lots, fillGuess, marginCheck))
      marginCheck = (leverage > 0) ? lots * spec.contract * SymbolInfoDouble(sym, SYMBOL_ASK) / leverage
                                    : lots * spec.contract * SymbolInfoDouble(sym, SYMBOL_ASK);
   if(!g_risk.MarginOk(marginCheck))
   {
      g_log.Log("INFO", "signal", sym, "rejected", PFX_KVs("reason", "margin"));
      return;
   }

   double tp = (g_sp.tpR > 0.0) ? fillGuess + sig.dir * g_sp.tpR * D : 0.0;
   sl = PFX_NormPrice(sym, sl);
   tp = (tp > 0.0) ? PFX_NormPrice(sym, tp) : 0.0;

   g_log.Log("INFO", "signal", sym, "taken",
            PFX_Join(PFX_KVs("signal_id", sig.id), PFX_Join(PFX_KVs("reason_text", sig.reason),
            PFX_Join(PFX_KV("atr_pips", sig.atr / spec.pip, 1), PFX_KV("lots", lots, 2)))));

   SOrderResult res;
   string comment = sig.dir > 0 ? "L_" : "S_";
   comment += TimeToString(sig.barTime, TIME_DATE) ;
   if(g_tm.Open(sym, sig.dir, lots, sl, tp, comment, res) && res.ok)
      g_risk.RegisterTradeOpened();
}

//--- direction-aware exposure count vs OPEN positions (excludes the symbol being considered)
int CurrencyDirCount(const string sym, const int dir, const bool baseSide)
{
   string names[]; int dirs[];
   g_tm.CurrencyExposure(sym, names, dirs);
   string ccy = baseSide ? SymbolInfoString(sym, SYMBOL_CURRENCY_BASE) : SymbolInfoString(sym, SYMBOL_CURRENCY_PROFIT);
   int wantDir = baseSide ? dir : -dir;
   return g_tm.ExpCount(names, dirs, ccy, wantDir);
}

//--- causal signal at the close of the bar that just closed (shift 1)
bool ComputeSignal(const string sym, const int idx, SSignal &out)
{
   out.valid = false;
   double atr;
   if(!g_ind[idx].Atr(1, atr)) return false;
   double pip = PFX_PipSize(sym);
   double atrPips = atr / pip;
   if(atrPips < g_sp.minAtrPips || atrPips > g_sp.maxAtrPips) return false;

   datetime barOpen = iTime(sym, g_sp.tfExec, 1);
   MqlDateTime dt; PFX_UtcStruct(barOpen + PeriodSeconds(g_sp.tfExec), dt); // hour of the NEXT (entry) bar, UTC
   int hourNext = dt.hour, wdNext = dt.day_of_week;
   if(wdNext == 0 || wdNext == 6) return false;
   if(hourNext < g_sp.sessionStartUtc || hourNext >= g_sp.sessionEndUtc) return false;
   if(wdNext == 5 && hourNext >= g_sp.fridayLastEntryUtc) return false;

   double o = iOpen(sym, g_sp.tfExec, 1), h = iHigh(sym, g_sp.tfExec, 1), l = iLow(sym, g_sp.tfExec, 1), c = iClose(sym, g_sp.tfExec, 1);
   double rng = h - l;
   if(rng > g_sp.maxBarRangeAtr * atr) return false;
   double clv = (rng > 0) ? (c - l) / rng : 0.5;

   double hh, ll;
   if(!g_ind[idx].Donchian(1, hh, ll)) return false;
   int trend;
   if(!g_ind[idx].Trend(barOpen, trend)) return false;

   int dir = 0;
   double level = 0;
   if(g_sp.allowLong && c > hh + g_sp.breakoutBufferAtr * atr && clv >= g_sp.minClv && trend == 1) { dir = 1; level = hh; }
   else if(g_sp.allowShort && c < ll - g_sp.breakoutBufferAtr * atr && clv <= 1.0 - g_sp.minClv && trend == -1) { dir = -1; level = ll; }
   if(dir == 0) return false;

   out.valid = true; out.dir = dir; out.barTime = barOpen; out.atr = atr; out.level = level;
   out.close = c; out.clv = clv; out.rangeAtr = rng / atr; out.trend = trend;
   out.id = sym + "-" + TimeToString(barOpen, TIME_DATE | TIME_MINUTES) + "-" + (dir > 0 ? "buy" : "sell");
   out.reason = StringFormat("close %s %s %s; trend %d; clv %.2f; atr %.1fp",
      DoubleToString(c, (int)SymbolInfoInteger(sym, SYMBOL_DIGITS)), (dir > 0 ? ">" : "<"),
      DoubleToString(level, (int)SymbolInfoInteger(sym, SYMBOL_DIGITS)), trend, clv, atrPips);
   return true;
}

//======================================================== management
void ManageOpenPosition(const string sym, const int idx, const bool mustFlatten, const string flattenReason)
{
   ulong ticket = 0; bool found = false;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) == sym && (long)PositionGetInteger(POSITION_MAGIC) == InpMagic)
      { ticket = tk; found = true; break; }
   }
   if(!found) return;
   if(!PositionSelectByTicket(ticket)) return;

   long   type   = PositionGetInteger(POSITION_TYPE);
   int    side   = (type == POSITION_TYPE_BUY) ? 1 : -1;
   double entry  = PositionGetDouble(POSITION_PRICE_OPEN);
   double sl     = PositionGetDouble(POSITION_SL);
   double vol    = PositionGetDouble(POSITION_VOLUME);
   double curPx  = (side > 0) ? SymbolInfoDouble(sym, SYMBOL_BID) : SymbolInfoDouble(sym, SYMBOL_ASK);

   if(mustFlatten)
   {
      PFX_SetExitReason(sym, flattenReason);
      g_tm.Close(ticket, flattenReason);
      g_lastExitTime[idx] = TimeTradeServer();
      PFX_GvSet(PFX_Key(InpMagic, "last_exit", sym), (double)g_lastExitTime[idx]);
      return;
   }

   // weekend flatten
   MqlDateTime dt; PFX_UtcStruct(TimeTradeServer(), dt);
   if(g_sp.closeBeforeWeekend && dt.day_of_week == 5 && dt.hour >= g_sp.weekendCloseHourUtc)
   {
      PFX_SetExitReason(sym, "weekend_close");
      g_tm.Close(ticket, "weekend_close");
      g_lastExitTime[idx] = TimeTradeServer();
      PFX_GvSet(PFX_Key(InpMagic, "last_exit", sym), (double)g_lastExitTime[idx]);
      return;
   }
   if(sl <= 0.0) return;   // never manage a position without a stop (shouldn't happen; defensive)

   double R = MathAbs(entry - sl);
   if(R <= 0.0) return;
   double fav = side * (curPx - entry);

   // break-even (once favourable move >= beR * R)
   string beKey = PFX_Key(InpMagic, "be_done", sym + "_" + (string)ticket);
   if(g_sp.beR > 0.0 && PFX_GvGet(beKey, 0) < 0.5 && fav >= g_sp.beR * R)
   {
      double cand = entry + side * g_sp.beOffsetPips * PFX_PipSize(sym);
      if((side > 0 && cand > sl) || (side < 0 && cand < sl))
      {
         if(g_tm.ModifyStop(ticket, PFX_NormPrice(sym, cand), PositionGetDouble(POSITION_TP)))
            sl = cand;
      }
      PFX_GvSet(beKey, 1);
   }

   // trailing stop (ATR-based, only ever tightens - ModifyStop refuses loosening)
   if(g_sp.trailStartR > 0.0 && fav >= g_sp.trailStartR * R)
   {
      double atrNow;
      if(g_ind[idx].Atr(0, atrNow) && atrNow > 0.0)
      {
         double cand = curPx - side * g_sp.trailAtrMult * atrNow;
         if((side > 0 && cand > sl) || (side < 0 && cand < sl))
            g_tm.ModifyStop(ticket, PFX_NormPrice(sym, cand), PositionGetDouble(POSITION_TP));
      }
   }

   // partial close
   string partKey = PFX_Key(InpMagic, "partial_done", sym + "_" + (string)ticket);
   if(g_sp.partialPct > 0.0 && PFX_GvGet(partKey, 0) < 0.5 && fav >= g_sp.partialR * R)
   {
      SSpec spec;
      if(PFX_LoadSpec(sym, spec))
      {
         double part = PFX_FloorLots(spec, vol * g_sp.partialPct);
         if(part >= spec.volMin - 1e-9 && (vol - part) >= spec.volMin - 1e-9)
            g_tm.Close(ticket, "partial", part);
      }
      PFX_GvSet(partKey, 1);
   }

   // time-in-trade exit and trend-flip exit act at the NEXT new bar via the normal position scan;
   // implemented here (checked every tick) for prompt response, still never touching SL (fills at market)
   if(g_sp.maxBarsInTrade > 0)
   {
      int barsHeld = iBarShift(sym, g_sp.tfExec, PositionGetInteger(POSITION_TIME), true);
      if(barsHeld >= g_sp.maxBarsInTrade)
      {
         PFX_SetExitReason(sym, "time_exit");
         g_tm.Close(ticket, "time_exit");
         g_lastExitTime[idx] = TimeTradeServer();
         PFX_GvSet(PFX_Key(InpMagic, "last_exit", sym), (double)g_lastExitTime[idx]);
         return;
      }
   }
}

//--- called once per new bar (from TryEnter's sibling path) to check trend flip on OPEN positions
void CheckTrendFlipOnNewBar(const string sym, const int idx)
{
   if(!g_sp.exitOnTrendFlip) return;
   ulong ticket = 0; bool found = false;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk != 0 && PositionGetString(POSITION_SYMBOL) == sym && (long)PositionGetInteger(POSITION_MAGIC) == InpMagic)
      { ticket = tk; found = true; break; }
   }
   if(!found) return;
   int side = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
   int trend;
   if(!g_ind[idx].Trend(iTime(sym, g_sp.tfExec, 1), trend)) return;
   if(trend == -side)
   {
      PFX_SetExitReason(sym, "trend_flip");
      g_tm.Close(ticket, "trend_flip");
      g_lastExitTime[idx] = TimeTradeServer();
      PFX_GvSet(PFX_Key(InpMagic, "last_exit", sym), (double)g_lastExitTime[idx]);
   }
}

//======================================================== trade events
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
   if(!HistoryDealSelect(trans.deal)) return;
   if((long)HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != InpMagic) return;
   string sym = HistoryDealGetString(trans.deal, DEAL_SYMBOL);
   long   entryType = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   if(entryType != DEAL_ENTRY_OUT && entryType != DEAL_ENTRY_OUT_BY) return;   // only log CLOSES here
   double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT) + HistoryDealGetDouble(trans.deal, DEAL_SWAP) +
                   HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);
   string reason = PFX_TakeExitReason(sym);
   if(reason == "") reason = DealReasonToStr((ENUM_DEAL_REASON)HistoryDealGetInteger(trans.deal, DEAL_REASON));
   g_log.Log("INFO", "trade_closed", sym, "closed",
            PFX_Join(PFX_KVs("exit_reason", reason), PFX_Join(PFX_KV("profit", profit, 2),
            PFX_Join(PFX_KV("volume", HistoryDealGetDouble(trans.deal, DEAL_VOLUME), 2),
                     PFX_KVi("deal", (long)trans.deal)))));
}

string DealReasonToStr(const ENUM_DEAL_REASON r)
{
   switch(r)
   {
      case DEAL_REASON_SL:       return "sl";
      case DEAL_REASON_TP:       return "tp";
      case DEAL_REASON_SO:       return "stop_out";
      case DEAL_REASON_EXPERT:   return "expert";
      case DEAL_REASON_CLIENT:   return "manual";
      case DEAL_REASON_MOBILE:   return "manual_mobile";
      case DEAL_REASON_WEB:      return "manual_web";
      default:                   return "other";
   }
}
