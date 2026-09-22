//+------------------------------------------------------------------+
//| Config.mqh - build SStrategyParams/SRiskParams/SExecParams from    |
//| the EA's `input` variables. DEFAULTS MUST MATCH                    |
//| config/strategy.toml and python/profx/config.py exactly.            |
//+------------------------------------------------------------------+
#ifndef PROFX_CONFIG_MQH
#define PROFX_CONFIG_MQH

#include "Defs.mqh"

SStrategyParams PFX_BuildStrategyParams(
   ENUM_TIMEFRAMES tfExec, ENUM_TIMEFRAMES tfTrend,
   int donchianN, int atrPeriod, int emaFast, int emaSlow, int slopeBars,
   double breakoutBufferAtr, double minClv, double maxBarRangeAtr,
   double minAtrPips, double maxAtrPips,
   int sessionStartUtc, int sessionEndUtc, int fridayLastEntryUtc,
   bool allowLong, bool allowShort,
   double slAtrMult, double tpR, double beR, double beOffsetPips,
   double trailStartR, double trailAtrMult, double partialR, double partialPct,
   int maxBarsInTrade, bool exitOnTrendFlip, bool closeBeforeWeekend, int weekendCloseHourUtc,
   int maxEntryDelaySec)
{
   SStrategyParams p;
   p.tfExec = tfExec; p.tfTrend = tfTrend;
   p.donchianN = donchianN; p.atrPeriod = atrPeriod; p.emaFast = emaFast; p.emaSlow = emaSlow;
   p.slopeBars = slopeBars; p.breakoutBufferAtr = breakoutBufferAtr; p.minClv = minClv;
   p.maxBarRangeAtr = maxBarRangeAtr; p.minAtrPips = minAtrPips; p.maxAtrPips = maxAtrPips;
   p.sessionStartUtc = sessionStartUtc; p.sessionEndUtc = sessionEndUtc; p.fridayLastEntryUtc = fridayLastEntryUtc;
   p.allowLong = allowLong; p.allowShort = allowShort;
   p.slAtrMult = slAtrMult; p.tpR = tpR; p.beR = beR; p.beOffsetPips = beOffsetPips;
   p.trailStartR = trailStartR; p.trailAtrMult = trailAtrMult; p.partialR = partialR; p.partialPct = partialPct;
   p.maxBarsInTrade = maxBarsInTrade; p.exitOnTrendFlip = exitOnTrendFlip;
   p.closeBeforeWeekend = closeBeforeWeekend; p.weekendCloseHourUtc = weekendCloseHourUtc;
   p.maxEntryDelaySec = maxEntryDelaySec;
   return p;
}

SRiskParams PFX_BuildRiskParams(
   double riskPct, bool riskOnEquity, int maxOpenPositions, double maxTotalRiskPct,
   int maxDailyTrades, int cooldownBars, double dailyLossLimitPct, double maxDrawdownPct,
   int maxSameCcyDir, double maxSpreadPips, double maxSpreadFracOfSl, double minFreeMarginPct,
   double commissionPerLotRt, bool flattenOnDailyLoss)
{
   SRiskParams r;
   r.riskPct = riskPct; r.riskOnEquity = riskOnEquity; r.maxOpenPositions = maxOpenPositions;
   r.maxTotalRiskPct = maxTotalRiskPct; r.maxDailyTrades = maxDailyTrades; r.cooldownBars = cooldownBars;
   r.dailyLossLimitPct = dailyLossLimitPct; r.maxDrawdownPct = maxDrawdownPct;
   r.maxSameCcyDir = maxSameCcyDir; r.maxSpreadPips = maxSpreadPips; r.maxSpreadFracOfSl = maxSpreadFracOfSl;
   r.minFreeMarginPct = minFreeMarginPct; r.commissionPerLotRt = commissionPerLotRt;
   r.flattenOnDailyLoss = flattenOnDailyLoss;
   return r;
}

//--- fail fast on unsafe/inconsistent settings rather than trading with bad config
bool PFX_ValidateConfig(const SStrategyParams &sp, const SRiskParams &rp, string &err)
{
   if(sp.donchianN < 2) { err = "donchianN < 2"; return false; }
   if(sp.emaFast >= sp.emaSlow) { err = "emaFast must be < emaSlow"; return false; }
   if(sp.minAtrPips >= sp.maxAtrPips) { err = "minAtrPips must be < maxAtrPips"; return false; }
   if(sp.sessionStartUtc < 0 || sp.sessionEndUtc > 24 || sp.sessionStartUtc >= sp.sessionEndUtc)
      { err = "invalid session hours"; return false; }
   if(sp.slAtrMult <= 0.0) { err = "slAtrMult must be > 0"; return false; }
   if(sp.minClv < 0.5 || sp.minClv > 1.0) { err = "minClv must be in [0.5,1.0]"; return false; }
   if(rp.riskPct <= 0.0 || rp.riskPct > 2.0) { err = "riskPct must be in (0,2] - hard safety cap"; return false; }
   if(rp.maxTotalRiskPct < rp.riskPct) { err = "maxTotalRiskPct must be >= riskPct"; return false; }
   if(rp.maxOpenPositions < 1) { err = "maxOpenPositions must be >= 1"; return false; }
   if(rp.dailyLossLimitPct <= 0.0 || rp.dailyLossLimitPct >= 100.0) { err = "invalid dailyLossLimitPct"; return false; }
   if(rp.maxDrawdownPct <= 0.0 || rp.maxDrawdownPct >= 100.0) { err = "invalid maxDrawdownPct"; return false; }
   return true;
}

#endif
