//+------------------------------------------------------------------+
//| Logger.mqh - structured JSON-lines logging                        |
//| One JSON object per line -> MQL5/Files/ProFX/logs/*.jsonl         |
//+------------------------------------------------------------------+
#ifndef PROFX_LOGGER_MQH
#define PROFX_LOGGER_MQH

#include "Utils.mqh"

class CLogger
{
private:
   long   m_magic;
   bool   m_verbose;
   int    m_common;      // FILE_COMMON or 0
   string m_tf;

public:
   void Init(const long magic, const bool verbose, const bool useCommon, const string tf)
   {
      m_magic   = magic;
      m_verbose = verbose;
      m_common  = useCommon ? FILE_COMMON : 0;
      m_tf      = tf;
      FolderCreate("ProFX", m_common);
      FolderCreate("ProFX\\logs", m_common);
   }

   bool Verbose() const { return m_verbose; }

   //--- fields: pre-built JSON members without braces, e.g. "\"sl\":1.1,\"tp\":1.2"
   void Log(const string level, const string event, const string symbol,
            const string state, const string fields)
   {
      if(level == "DEBUG" && !m_verbose) return;
      datetime srv = TimeTradeServer();
      string line = "{" +
                    PFX_KVs("ts_utc", PFX_IsoUtc(srv)) + "," +
                    PFX_KVs("ts_server", TimeToString(srv, TIME_DATE | TIME_SECONDS)) + "," +
                    PFX_KVs("lvl", level) + "," +
                    PFX_KVs("event", event) + "," +
                    PFX_KVs("symbol", symbol) + "," +
                    PFX_KVs("tf", m_tf) + "," +
                    PFX_KVs("state", state) + "," +
                    PFX_KVi("magic", m_magic);
      if(fields != "") line += "," + fields;
      line += "}";
      Print(line);
      Write(line);
   }

private:
   void Write(const string line)
   {
      MqlDateTime d;
      TimeToStruct(TimeGMT(), d);
      string name = StringFormat("ProFX\\logs\\ProFX_%I64d_%04d%02d%02d.jsonl", m_magic, d.year, d.mon, d.day);
      int h = FileOpen(name, FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_SHARE_READ | FILE_SHARE_WRITE | m_common);
      if(h == INVALID_HANDLE) return;      // logging must never crash trading logic
      FileSeek(h, 0, SEEK_END);
      FileWriteString(h, line + "\r\n");
      FileClose(h);
   }
};

#endif
