//+------------------------------------------------------------------+
//| RiskManager.mqh - position sizing, exposure limits, circuit breakers|
//| No order sending here - this module only says YES/NO and HOW MUCH. |
//+------------------------------------------------------------------+
#ifndef PROFX_RISKMANAGER_MQH
#define PROFX_RISKMANAGER_MQH

#include "Defs.mqh"
#include "Utils.mqh"
#include "Logger.mqh"

class CRiskManager
{
private:
   SRiskParams m_r;
   long        m_magic;
   CLogger    *m_log;
   double      m_peakEquity;
   double      m_dayStartEquity;
   int         m_dayKey;
   int         m_dailyTrades;
   bool        m_dailyBlocked;
   bool        m_halted;
   string      m_haltReason;

public:
   void Init(const SRiskParams &r, const long magic, CLogger *log)
   {
      m_r = r; m_magic = magic; m_log = log;
      m_peakEquity     = PFX_GvGet(PFX_Key(magic, "peak_equity"), AccountInfoDouble(ACCOUNT_EQUITY));
      m_dayStartEquity = PFX_GvGet(PFX_Key(magic, "day_start_equity"), AccountInfoDouble(ACCOUNT_EQUITY));
      m_dayKey         = (int)PFX_GvGet(PFX_Key(magic, "day_key"), 0);
      m_dailyTrades    = (int)PFX_GvGet(PFX_Key(magic, "daily_trades"), 0);
      m_dailyBlocked   = PFX_GvGet(PFX_Key(magic, "daily_blocked"), 0) > 0.5;
      m_halted         = PFX_GvGet(PFX_Key(magic, "halted"), 0) > 0.5;
      m_haltReason     = "";
   }

   bool Halted() const { return m_halted; }
   bool DailyBlocked() const { return m_dailyBlocked; }

   //--- call once per new exec bar (and at OnInit) before any trading decision
   void OnNewBar()
   {
      MqlDateTime dt;
      PFX_UtcStruct(TimeTradeServer(), dt);
      int key = dt.year * 10000 + dt.mon * 100 + dt.day;
      if(key != m_dayKey)
      {
         m_dayKey         = key;
         m_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
         m_dailyTrades    = 0;
         m_dailyBlocked   = false;
         Persist();
      }
      double eq = AccountInfoDouble(ACCOUNT_EQUITY);
      if(eq > m_peakEquity) { m_peakEquity = eq; Persist(); }

      double ddPct = (m_peakEquity > 0) ? (m_peakEquity - eq) / m_peakEquity * 100.0 : 0.0;
      if(!m_halted && ddPct >= m_r.maxDrawdownPct)
      {
         m_halted = true; m_haltReason = "max_drawdown";
         Persist();
         if(m_log != NULL)
            m_log.Log("ERROR", "circuit_breaker", "*", "halted",
                      PFX_Join(PFX_KVs("reason", "max_drawdown"), PFX_KV("dd_pct", ddPct, 2)));
      }
      double dayRetPct = (m_dayStartEquity > 0) ? (eq / m_dayStartEquity - 1.0) * 100.0 : 0.0;
      if(!m_dailyBlocked && dayRetPct <= -m_r.dailyLossLimitPct)
      {
         m_dailyBlocked = true;
         Persist();
         if(m_log != NULL)
            m_log.Log("ERROR", "circuit_breaker", "*", "daily_loss_block",
                      PFX_KV("day_ret_pct", dayRetPct, 2));
      }
   }

   //--- must-flatten-everything check (halted, or daily block with flatten enabled)
   bool MustFlattenAll(string &reason) const
   {
      if(m_halted) { reason = "max_drawdown"; return true; }
      if(m_dailyBlocked && m_r.flattenOnDailyLoss) { reason = "daily_loss_limit"; return true; }
      return false;
   }

   //--- manual/admin reset (NOT automatic - operator must confirm root cause was fixed)
   void ResetHalt()
   {
      m_halted = false; m_haltReason = ""; m_peakEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      Persist();
   }

   void RegisterTradeOpened() { m_dailyTrades++; Persist(); }

   //--- position sizing: lots so a stop-out loses <= riskPct of equity/balance, NEVER rounds up
   double CalcLots(const SSpec &spec, const double entryPrice, const double slPrice)
   {
      double dist = MathAbs(entryPrice - slPrice);
      if(dist <= 0.0) return 0.0;
      double basis = m_r.riskOnEquity ? AccountInfoDouble(ACCOUNT_EQUITY) : AccountInfoDouble(ACCOUNT_BALANCE);
      double lossPerLot = dist / spec.tickSize * spec.tickValueLoss + m_r.commissionPerLotRt;
      if(lossPerLot <= 0.0) return 0.0;
      double raw = basis * m_r.riskPct / 100.0 / lossPerLot;
      return PFX_FloorLots(spec, raw);
   }

   double RiskMoney(const SSpec &spec, const double lots, const double entryPrice, const double slPrice)
   {
      double dist = MathAbs(entryPrice - slPrice);
      return lots * (dist / spec.tickSize * spec.tickValueLoss + m_r.commissionPerLotRt);
   }

   //--- gate checks. Returns "" if OK, else a rejection reason string.
   string CheckEntryGates(const string sym, const int openPositionsTotal, const int openPositionsThisSymbol,
                          const datetime lastExitTime, const double spreadPrice, const double slDistPrice,
                          const int sameBaseSameDir, const int sameQuoteOppDir)
   {
      if(m_halted) return "halted";
      if(m_dailyBlocked) return "daily_loss_block";
      if(openPositionsThisSymbol > 0) return "symbol_busy";
      if(openPositionsTotal >= m_r.maxOpenPositions) return "max_positions";
      if(m_dailyTrades >= m_r.maxDailyTrades) return "max_daily_trades";
      if(lastExitTime > 0)
      {
         int barSec = PeriodSeconds(PERIOD_H1);
         if((long)TimeTradeServer() - (long)lastExitTime < (long)m_r.cooldownBars * barSec) return "cooldown";
      }
      double pip = PFX_PipSize(sym);
      if(spreadPrice / pip > m_r.maxSpreadPips) return "spread_abs";
      if(spreadPrice > m_r.maxSpreadFracOfSl * slDistPrice) return "spread_rel";
      if(sameBaseSameDir + 1 > m_r.maxSameCcyDir) return "currency_exposure";
      if(sameQuoteOppDir + 1 > m_r.maxSameCcyDir) return "currency_exposure";
      return "";
   }

   bool TotalRiskOk(const double openRiskMoney, const double newRiskMoney)
   {
      double eq = AccountInfoDouble(ACCOUNT_EQUITY);
      return (openRiskMoney + newRiskMoney) <= m_r.maxTotalRiskPct / 100.0 * eq;
   }

   bool MarginOk(const double marginRequired)
   {
      double eq = AccountInfoDouble(ACCOUNT_EQUITY);
      double freeAfter = eq - marginRequired;
      return freeAfter >= m_r.minFreeMarginPct / 100.0 * eq;
   }

private:
   void Persist()
   {
      PFX_GvSet(PFX_Key(m_magic, "peak_equity"), m_peakEquity);
      PFX_GvSet(PFX_Key(m_magic, "day_start_equity"), m_dayStartEquity);
      PFX_GvSet(PFX_Key(m_magic, "day_key"), m_dayKey);
      PFX_GvSet(PFX_Key(m_magic, "daily_trades"), m_dailyTrades);
      PFX_GvSet(PFX_Key(m_magic, "daily_blocked"), m_dailyBlocked ? 1 : 0);
      PFX_GvSet(PFX_Key(m_magic, "halted"), m_halted ? 1 : 0);
   }
};

#endif
