//+------------------------------------------------------------------+
//| Utils.mqh - small, dependency-free helpers                        |
//+------------------------------------------------------------------+
#ifndef PROFX_UTILS_MQH
#define PROFX_UTILS_MQH

#include "Defs.mqh"

double PFX_PipSize(const string sym)
{
   int    d  = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   double pt = SymbolInfoDouble(sym, SYMBOL_POINT);
   return (d == 3 || d == 5) ? pt * 10.0 : pt;
}

double PFX_NormPrice(const string sym, const double p)
{
   double ts = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   int    d  = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   if(ts <= 0.0) return NormalizeDouble(p, d);
   return NormalizeDouble(MathRound(p / ts) * ts, d);
}

//--- server->UTC offset, rounded to 15 minutes (works in tester: TimeGMT is simulated there)
long PFX_UtcOffsetSec()
{
   long off = (long)(TimeTradeServer() - TimeGMT());
   long q   = (off >= 0) ? 450 : -450;
   return ((off + q) / 900) * 900;
}

datetime PFX_ServerToUtc(const datetime t)
{
   return (datetime)((long)t - PFX_UtcOffsetSec());
}

void PFX_UtcStruct(const datetime serverTime, MqlDateTime &dt)
{
   TimeToStruct(PFX_ServerToUtc(serverTime), dt);
}

string PFX_IsoUtc(const datetime serverTime)
{
   string s = TimeToString(PFX_ServerToUtc(serverTime), TIME_DATE | TIME_SECONDS);
   StringReplace(s, ".", "-");
   StringReplace(s, " ", "T");
   return s + "Z";
}

string PFX_Esc(const string s)
{
   string r = s;
   StringReplace(r, "\\", "\\\\");
   StringReplace(r, "\"", "'");
   StringReplace(r, "\r", " ");
   StringReplace(r, "\n", " ");
   return r;
}

//--- JSON fragment builders
string PFX_KV(const string k, const double v, const int digits = 5)
{
   return "\"" + k + "\":" + DoubleToString(v, digits);
}
string PFX_KVi(const string k, const long v)
{
   return "\"" + k + "\":" + IntegerToString(v);
}
string PFX_KVs(const string k, const string v)
{
   return "\"" + k + "\":\"" + PFX_Esc(v) + "\"";
}
string PFX_KVb(const string k, const bool v)
{
   return "\"" + k + "\":" + (v ? "true" : "false");
}
string PFX_Join(const string a, const string b)
{
   if(a == "") return b;
   if(b == "") return a;
   return a + "," + b;
}

//--- global-variable persistence (survives terminal restarts) -------
string PFX_Key(const long magic, const string name, const string sym = "")
{
   string k = "PFX_" + (string)magic + "_" + name;
   if(sym != "") k += "_" + sym;
   return k;
}
double PFX_GvGet(const string k, const double def)
{
   if(GlobalVariableCheck(k)) return GlobalVariableGet(k);
   return def;
}
void PFX_GvSet(const string k, const double v)
{
   GlobalVariableSet(k, v);
}
void PFX_GvDel(const string k)
{
   if(GlobalVariableCheck(k)) GlobalVariableDel(k);
}

//--- exit-reason bookkeeping (EA-initiated closes) -------------------
string g_pfxExitSym[];
string g_pfxExitReason[];

void PFX_SetExitReason(const string sym, const string reason)
{
   int n = ArraySize(g_pfxExitSym);
   for(int i = 0; i < n; i++)
      if(g_pfxExitSym[i] == sym) { g_pfxExitReason[i] = reason; return; }
   ArrayResize(g_pfxExitSym, n + 1);
   ArrayResize(g_pfxExitReason, n + 1);
   g_pfxExitSym[n]    = sym;
   g_pfxExitReason[n] = reason;
}

string PFX_TakeExitReason(const string sym)
{
   int n = ArraySize(g_pfxExitSym);
   for(int i = 0; i < n; i++)
      if(g_pfxExitSym[i] == sym)
      {
         string r = g_pfxExitReason[i];
         g_pfxExitReason[i] = "";
         return r;
      }
   return "";
}

//--- symbol specification ---------------------------------------------
bool PFX_LoadSpec(const string sym, SSpec &s)
{
   s.sym           = sym;
   s.base          = SymbolInfoString(sym, SYMBOL_CURRENCY_BASE);
   s.quote         = SymbolInfoString(sym, SYMBOL_CURRENCY_PROFIT);
   s.digits        = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   s.point         = SymbolInfoDouble(sym, SYMBOL_POINT);
   s.pip           = PFX_PipSize(sym);
   s.tickSize      = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   s.tickValueLoss = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE_LOSS);
   if(s.tickValueLoss <= 0.0) s.tickValueLoss = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   s.contract      = SymbolInfoDouble(sym, SYMBOL_TRADE_CONTRACT_SIZE);
   s.volMin        = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   s.volMax        = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);
   s.volStep       = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   s.stopsLevel    = (int)SymbolInfoInteger(sym, SYMBOL_TRADE_STOPS_LEVEL);
   s.freezeLevel   = (int)SymbolInfoInteger(sym, SYMBOL_TRADE_FREEZE_LEVEL);
   return (s.point > 0.0 && s.tickSize > 0.0 && s.tickValueLoss > 0.0 &&
           s.volStep > 0.0 && s.volMin > 0.0 && s.volMax > 0.0 && s.digits > 0);
}

//--- floor to lot step; NEVER round up (that would exceed the risk budget)
double PFX_FloorLots(const SSpec &s, const double lots)
{
   if(!MathIsValidNumber(lots) || lots <= 0.0) return 0.0;
   double v = MathFloor(lots / s.volStep + 1e-9) * s.volStep;
   v = MathMin(v, s.volMax);
   return NormalizeDouble(v, 8);
}

#endif
