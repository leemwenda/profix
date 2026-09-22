//+------------------------------------------------------------------+
//| TradeManager.mqh - order execution and in-trade management         |
//| Only ever touches positions with THIS EA's magic number.            |
//+------------------------------------------------------------------+
#ifndef PROFX_TRADEMANAGER_MQH
#define PROFX_TRADEMANAGER_MQH

#include <Trade\Trade.mqh>
#include "Defs.mqh"
#include "Utils.mqh"
#include "Logger.mqh"
#include "RiskManager.mqh"

class CTradeManager
{
private:
   CTrade        m_trade;
   SExecParams   m_e;
   CLogger      *m_log;
   datetime      m_lastOrderTime;
   int           m_ordersThisHour;
   datetime      m_hourKey;
   int           m_consecReject;

public:
   void Init(const SExecParams &e, CLogger *log)
   {
      m_e = e; m_log = log;
      m_trade.SetExpertMagicNumber(e.magic);
      m_trade.SetDeviationInPoints(e.deviationPoints);
      m_trade.SetTypeFillingBySymbol(_Symbol);
      m_trade.SetAsyncMode(false);
      m_lastOrderTime = 0; m_ordersThisHour = 0; m_hourKey = 0; m_consecReject = 0;
   }

   bool TooManyConsecutiveRejects() const { return m_consecReject >= m_e.maxConsecutiveRejects; }

   //--- true if it is currently safe (rate-limit-wise) to send a new order
   bool RateLimitOk()
   {
      datetime now = TimeTradeServer();
      if(m_lastOrderTime > 0 && (long)now - (long)m_lastOrderTime < m_e.minSecondsBetweenOrders) return false;
      datetime hk = (datetime)(((long)now) / 3600 * 3600);
      if(hk != m_hourKey) { m_hourKey = hk; m_ordersThisHour = 0; }
      return m_ordersThisHour < m_e.maxOrdersPerHour;
   }

   //--- true if this symbol already has a position with our magic (never double-enter / never hedge)
   bool HasOpenPosition(const string sym) const
   {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         ulong tk = PositionGetTicket(i);
         if(tk == 0) continue;
         if(PositionGetString(POSITION_SYMBOL) == sym && (long)PositionGetInteger(POSITION_MAGIC) == m_e.magic)
            return true;
      }
      return false;
   }

   int CountOpenPositions() const
   {
      int n = 0;
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         ulong tk = PositionGetTicket(i);
         if(tk != 0 && (long)PositionGetInteger(POSITION_MAGIC) == m_e.magic) n++;
      }
      return n;
   }

   double SumOpenRiskMoney(const CRiskManager &rm, const string excludeSym = "") const
   {
      double tot = 0.0;
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         ulong tk = PositionGetTicket(i);
         if(tk == 0) continue;
         if((long)PositionGetInteger(POSITION_MAGIC) != m_e.magic) continue;
         string sym = PositionGetString(POSITION_SYMBOL);
         if(sym == excludeSym) continue;
         double entry = PositionGetDouble(POSITION_PRICE_OPEN);
         double sl    = PositionGetDouble(POSITION_SL);
         double vol   = PositionGetDouble(POSITION_VOLUME);
         SSpec sp;
         if(sl > 0.0 && PFX_LoadSpec(sym, sp))
            tot += vol * MathAbs(entry - sl) / sp.tickSize * sp.tickValueLoss;
      }
      return tot;
   }

   //--- +1/-1 exposure counters per currency, EXCLUDING excludeSym (used for the pre-trade check)
   void CurrencyExposure(const string excludeSym, string &names[], int &dirs[])
   {
      ArrayResize(names, 0); ArrayResize(dirs, 0);
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         ulong tk = PositionGetTicket(i);
         if(tk == 0) continue;
         if((long)PositionGetInteger(POSITION_MAGIC) != m_e.magic) continue;
         string sym = PositionGetString(POSITION_SYMBOL);
         if(sym == excludeSym) continue;
         int side = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
         string b = SymbolInfoString(sym, SYMBOL_CURRENCY_BASE);
         string q = SymbolInfoString(sym, SYMBOL_CURRENCY_PROFIT);
         AddExp(names, dirs, b, side);
         AddExp(names, dirs, q, -side);
      }
   }

   int ExpCount(const string &names[], const int &dirs[], const string ccy, const int dir)
   {
      return DirCount(names, dirs, ccy, dir);
   }

   //--- open a position; verifies AFTER the call that it actually exists before counting it as opened
   bool Open(const string sym, const int dir, const double lots, const double sl, const double tp,
            const string comment, SOrderResult &out)
   {
      out.ok = false;
      if(m_e.dryRun)
      {
         out.ok = true; out.price = 0; out.volume = lots; out.sl = sl; out.tp = tp;
         out.comment = "DRYRUN:" + comment;
         LogResult("dry_run_order", sym, out, dir);
         return true;
      }
      if(HasOpenPosition(sym)) { out.comment = "duplicate_guard"; return false; }  // last-line duplicate guard
      if(!RateLimitOk()) { out.comment = "rate_limited"; return false; }

      bool sent = false;
      double price = 0.0;
      for(int attempt = 0; attempt < MathMax(1, m_e.maxRetries); attempt++)
      {
         MqlTick tick;
         if(!SymbolInfoTick(sym, tick)) { Sleep(200); continue; }
         if(m_e.maxTickAgeSec > 0 && (long)TimeTradeServer() - (long)tick.time > m_e.maxTickAgeSec) { Sleep(200); continue; }
         price = (dir > 0) ? tick.ask : tick.bid;
         m_trade.SetDeviationInPoints(m_e.deviationPoints);
         bool ok = (dir > 0) ? m_trade.Buy(lots, sym, price, sl, tp, m_e.commentPrefix + comment)
                              : m_trade.Sell(lots, sym, price, sl, tp, m_e.commentPrefix + comment);
         uint rc = m_trade.ResultRetcode();
         if(ok && (rc == TRADE_RETCODE_DONE || rc == TRADE_RETCODE_PLACED))
         {
            sent = true;
            out.ticket = m_trade.ResultOrder();
            out.price  = m_trade.ResultPrice();
            break;
         }
         // requotes / off-quotes / timeouts are retryable; invalid volume/stops are not
         if(rc == TRADE_RETCODE_REQUOTE || rc == TRADE_RETCODE_PRICE_CHANGED || rc == TRADE_RETCODE_TIMEOUT ||
            rc == TRADE_RETCODE_CONNECTION || rc == TRADE_RETCODE_NO_MONEY)
         {
            if(m_log != NULL)
               m_log.Log("WARN", "order_retry", sym, "retry", PFX_Join(PFX_KVi("attempt", attempt),
                         PFX_KVi("retcode", (long)rc)));
            Sleep(500);
            continue;
         }
         out.retcode = rc; out.comment = m_trade.ResultComment();
         break;
      }
      m_lastOrderTime = TimeTradeServer();
      m_ordersThisHour++;
      out.retcode = m_trade.ResultRetcode();
      out.volume  = lots; out.sl = sl; out.tp = tp;
      if(sent)
      {
         // verify a real position now exists before trusting the fill (protects against desync)
         if(!HasOpenPosition(sym)) { out.ok = false; out.comment = "no_position_after_send"; }
         else { out.ok = true; m_consecReject = 0; }
      }
      else m_consecReject++;
      LogResult(sent ? "order_filled" : "order_failed", sym, out, dir);
      return out.ok;
   }

   //--- close (fully or partially) a position by ticket
   bool Close(const ulong ticket, const string reason, const double volume = 0.0)
   {
      if(!PositionSelectByTicket(ticket)) return false;
      string sym = PositionGetString(POSITION_SYMBOL);
      if(m_e.dryRun)
      {
         if(m_log != NULL) m_log.Log("INFO", "dry_run_close", sym, "close", PFX_KVs("reason", reason));
         return true;
      }
      double vol = (volume > 0.0) ? volume : PositionGetDouble(POSITION_VOLUME);
      bool ok = (volume > 0.0 && volume < PositionGetDouble(POSITION_VOLUME))
                ? m_trade.PositionClosePartial(ticket, vol)
                : m_trade.PositionClose(ticket);
      if(m_log != NULL)
         m_log.Log(ok ? "INFO" : "ERROR", "close", sym, ok ? "closed" : "close_failed",
                   PFX_Join(PFX_KVs("reason", reason), PFX_Join(PFX_KV("volume", vol, 2),
                            PFX_KVi("retcode", (long)m_trade.ResultRetcode()))));
      return ok;
   }

   //--- modify SL/TP; the caller (RiskManager-driven logic in the EA) must already guarantee
   //--- the new SL never increases risk. This function additionally refuses to WIDEN a stop away
   //--- from the market as a defence-in-depth check.
   bool ModifyStop(const ulong ticket, const double newSl, const double newTp)
   {
      if(!PositionSelectByTicket(ticket)) return false;
      long type = PositionGetInteger(POSITION_TYPE);
      double curSl = PositionGetDouble(POSITION_SL);
      if(curSl > 0.0)
      {
         if(type == POSITION_TYPE_BUY && newSl < curSl - _Point) return false;   // would loosen -> refuse
         if(type == POSITION_TYPE_SELL && newSl > curSl + _Point) return false;
      }
      if(m_e.dryRun) return true;
      bool ok = m_trade.PositionModify(PositionGetString(POSITION_SYMBOL), newSl, newTp);
      if(!ok && m_log != NULL)
         m_log.Log("WARN", "modify_failed", PositionGetString(POSITION_SYMBOL), "manage",
                  PFX_KVi("retcode", (long)m_trade.ResultRetcode()));
      return ok;
   }

private:
   void AddExp(string &names[], int &dirs[], const string ccy, const int dir)
   {
      int n = ArraySize(names);
      ArrayResize(names, n + 1); ArrayResize(dirs, n + 1);
      names[n] = ccy; dirs[n] = dir;
   }

   int DirCount(const string &names[], const int &dirs[], const string ccy, const int dir)
   {
      int c = 0;
      for(int i = 0; i < ArraySize(names); i++)
         if(names[i] == ccy && dirs[i] == dir) c++;
      return c;
   }

   void LogResult(const string event, const string sym, const SOrderResult &r, const int dir)
   {
      if(m_log == NULL) return;
      string f = PFX_Join(PFX_KVs("side", dir > 0 ? "buy" : "sell"),
                 PFX_Join(PFX_KV("volume", r.volume, 2),
                 PFX_Join(PFX_KV("price", r.price, 5),
                 PFX_Join(PFX_KV("sl", r.sl, 5),
                 PFX_Join(PFX_KV("tp", r.tp, 5),
                 PFX_Join(PFX_KVi("retcode", (long)r.retcode),
                          PFX_KVs("comment", r.comment)))))));
      m_log.Log(r.ok ? "INFO" : "ERROR", event, sym, r.ok ? "opened" : "rejected", f);
   }
};

#endif
