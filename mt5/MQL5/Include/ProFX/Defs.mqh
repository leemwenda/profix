//+------------------------------------------------------------------+
//| Defs.mqh - shared types (no logic)                                |
//+------------------------------------------------------------------+
#ifndef PROFX_DEFS_MQH
#define PROFX_DEFS_MQH

#define PFX_VERSION "1.0.0"

enum ENUM_PFX_MODE
{
   PFX_MODE_DRYRUN = 0,   // DRY-RUN: evaluate + log, NEVER send orders (live/demo terminal)
   PFX_MODE_TRADE  = 1    // TRADE: send orders (also requires InpConfirmLive=true)
};

struct SStrategyParams
{
   ENUM_TIMEFRAMES tfExec;
   ENUM_TIMEFRAMES tfTrend;
   int    donchianN;
   int    atrPeriod;
   int    emaFast;
   int    emaSlow;
   int    slopeBars;
   double breakoutBufferAtr;
   double minClv;
   double maxBarRangeAtr;
   double minAtrPips;
   double maxAtrPips;
   int    sessionStartUtc;
   int    sessionEndUtc;
   int    fridayLastEntryUtc;
   bool   allowLong;
   bool   allowShort;
   double slAtrMult;
   double tpR;
   double beR;
   double beOffsetPips;
   double trailStartR;
   double trailAtrMult;
   double partialR;
   double partialPct;
   int    maxBarsInTrade;
   bool   exitOnTrendFlip;
   bool   closeBeforeWeekend;
   int    weekendCloseHourUtc;
   int    maxEntryDelaySec;
};

struct SRiskParams
{
   double riskPct;
   bool   riskOnEquity;
   int    maxOpenPositions;
   double maxTotalRiskPct;
   int    maxDailyTrades;
   int    cooldownBars;
   double dailyLossLimitPct;
   double maxDrawdownPct;
   int    maxSameCcyDir;
   double maxSpreadPips;
   double maxSpreadFracOfSl;
   double minFreeMarginPct;
   double commissionPerLotRt;
   bool   flattenOnDailyLoss;
};

struct SExecParams
{
   long   magic;
   int    deviationPoints;
   int    maxRetries;
   int    minSecondsBetweenOrders;
   int    maxOrdersPerHour;
   int    maxConsecutiveRejects;
   bool   dryRun;
   int    maxTickAgeSec;
   string commentPrefix;
};

struct SSignal
{
   bool     valid;
   int      dir;        // +1 buy / -1 sell
   datetime barTime;    // open time (server) of the signal bar (shift 1)
   double   atr;
   double   level;      // breakout level (HH or LL)
   double   close;
   double   clv;
   double   rangeAtr;
   int      trend;
   string   reason;
   string   id;
};

struct SOrderResult
{
   bool   ok;
   uint   retcode;
   ulong  ticket;
   double price;
   double volume;
   double sl;
   double tp;
   string comment;
};

struct SSpec
{
   string sym;
   string base;
   string quote;
   int    digits;
   double point;
   double pip;
   double tickSize;
   double tickValueLoss;
   double contract;
   double volMin;
   double volMax;
   double volStep;
   int    stopsLevel;
   int    freezeLevel;
};

#endif
