//+------------------------------------------------------------------+
//| Indicators.mqh - causal indicator state, mirrors python/profx/signals.py |
//| Contract: a value at CLOSED bar shift k is computed ONLY from bars  |
//| >= k (older). Nothing here may reference shift 0 (the forming bar). |
//+------------------------------------------------------------------+
#ifndef PROFX_INDICATORS_MQH
#define PROFX_INDICATORS_MQH

#include "Defs.mqh"
#include "Utils.mqh"

class CIndicators
{
private:
   string m_sym;
   SStrategyParams m_p;
   int    m_hAtr, m_hEmaFastH, m_hEmaSlowH;
   bool   m_ready;

public:
   CIndicators() { m_hAtr = INVALID_HANDLE; m_hEmaFastH = INVALID_HANDLE; m_hEmaSlowH = INVALID_HANDLE; m_ready = false; }

   bool Init(const string sym, const SStrategyParams &p)
   {
      m_sym = sym; m_p = p;
      m_hAtr      = iATR(sym, p.tfExec, p.atrPeriod);
      m_hEmaFastH = iMA(sym, p.tfTrend, p.emaFast, 0, MODE_EMA, PRICE_CLOSE);
      m_hEmaSlowH = iMA(sym, p.tfTrend, p.emaSlow, 0, MODE_EMA, PRICE_CLOSE);
      m_ready = (m_hAtr != INVALID_HANDLE && m_hEmaFastH != INVALID_HANDLE && m_hEmaSlowH != INVALID_HANDLE);
      if(!m_ready)
         PrintFormat("ProFX Indicators: failed to create indicator handles for %s", sym);
      return m_ready;
   }

   ~CIndicators()
   {
      if(m_hAtr != INVALID_HANDLE) IndicatorRelease(m_hAtr);
      if(m_hEmaFastH != INVALID_HANDLE) IndicatorRelease(m_hEmaFastH);
      if(m_hEmaSlowH != INVALID_HANDLE) IndicatorRelease(m_hEmaSlowH);
   }

   bool Ready() const { return m_ready; }

   //--- ATR value of CLOSED bar `shift` (shift=1 is the most recently closed exec bar)
   bool Atr(const int shift, double &out)
   {
      double buf[];
      if(CopyBuffer(m_hAtr, 0, shift, 1, buf) != 1) return false;
      out = buf[0];
      return (out > 0.0 && MathIsValidNumber(out));
   }

   //--- highest high / lowest low of the N exec bars STRICTLY BEFORE `shift` (Donchian, causal)
   bool Donchian(const int shift, double &hh, double &ll)
   {
      int hIdx = iHighest(m_sym, m_p.tfExec, MODE_HIGH, m_p.donchianN, shift + 1);
      int lIdx = iLowest(m_sym, m_p.tfExec, MODE_LOW, m_p.donchianN, shift + 1);
      if(hIdx < 0 || lIdx < 0) return false;
      hh = iHigh(m_sym, m_p.tfExec, hIdx);
      ll = iLow(m_sym, m_p.tfExec, lIdx);
      return (hh > 0.0 && ll > 0.0);
   }

   //--- HTF trend using only a CLOSED trend-timeframe bar (never the forming one)
   //--- trend-bar close time must be <= signal bar's open time + exec period (matches python HTF gate)
   bool Trend(const datetime signalBarOpenServer, int &trend)
   {
      trend = 0;
      datetime trendBarOpen = iTime(m_sym, m_p.tfTrend, 0);
      int      trendPeriodSec = PeriodSeconds(m_p.tfTrend);
      // if the current (forming) trend bar's open is <= our signal bar open, the last CLOSED
      // trend bar is shift 1; otherwise shift 0 (already closed relative to the signal bar).
      int base = (trendBarOpen <= signalBarOpenServer) ? 1 : 0;
      double efNow, efPrev, es;
      double buf[];
      if(CopyBuffer(m_hEmaFastH, 0, base, 1, buf) != 1) return false;
      efNow = buf[0];
      if(CopyBuffer(m_hEmaFastH, 0, base + m_p.slopeBars, 1, buf) != 1) return false;
      efPrev = buf[0];
      if(CopyBuffer(m_hEmaSlowH, 0, base, 1, buf) != 1) return false;
      es = buf[0];
      if(!MathIsValidNumber(efNow) || !MathIsValidNumber(efPrev) || !MathIsValidNumber(es)) return false;
      if(efNow > es && efNow > efPrev) trend = 1;
      else if(efNow < es && efNow < efPrev) trend = -1;
      return true;
   }
};

#endif
