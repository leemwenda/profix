"""Settings come from ENVIRONMENT VARIABLES only. Nothing sensitive is ever in source code."""
from __future__ import annotations

import os
from dataclasses import dataclass


class SettingsError(RuntimeError):
    pass


def _b(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    secret: str                        # WEBHOOK_SECRET (>= 24 chars)
    allow_passphrase_auth: bool = True  # TradingView cannot sign requests; it can only embed a passphrase
    ttl_seconds: int = 120             # signal timestamp must be within +-ttl of server clock
    max_body_bytes: int = 4096
    allowed_symbols: tuple[str, ...] = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD")
    allowed_strategies: tuple[str, ...] = ("PROFX_TREND_V1",)
    allowed_timeframes: tuple[str, ...] = ("60", "1H", "H1")
    max_risk_pct: float = 0.5          # hard ceiling; client risk_pct is CLAMPED-REJECTED above this
    max_price_dev_pct: float = 0.15    # |signal price - live price| tolerance
    max_signals_per_minute: int = 10
    dry_run: bool = True               # WEBHOOK_LIVE=1 to disable
    allow_real_account: bool = False   # WEBHOOK_ALLOW_REAL=1 to permit non-demo accounts
    db_path: str = "webhook_state.sqlite3"
    kill_file: str = "PROFX_KILL.flag"
    magic: int = 26092101
    max_spread_pips: float = 3.0
    retention_days: int = 7

    @staticmethod
    def from_env() -> "Settings":
        secret = os.environ.get("WEBHOOK_SECRET", "")
        if len(secret) < 24:
            raise SettingsError("WEBHOOK_SECRET must be set (>= 24 random chars). Example: python -c \"import secrets;print(secrets.token_urlsafe(32))\"")
        sym = tuple(s.strip().upper() for s in os.environ.get("WEBHOOK_SYMBOLS", "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD").split(",") if s.strip())
        return Settings(
            secret=secret,
            allow_passphrase_auth=_b("WEBHOOK_ALLOW_PASSPHRASE", True),
            ttl_seconds=int(os.environ.get("WEBHOOK_TTL_SECONDS", "120")),
            allowed_symbols=sym,
            max_risk_pct=float(os.environ.get("WEBHOOK_MAX_RISK_PCT", "0.5")),
            dry_run=not _b("WEBHOOK_LIVE", False),
            allow_real_account=_b("WEBHOOK_ALLOW_REAL", False),
            db_path=os.environ.get("WEBHOOK_DB", "webhook_state.sqlite3"),
            kill_file=os.environ.get("WEBHOOK_KILL_FILE", "PROFX_KILL.flag"),
            magic=int(os.environ.get("WEBHOOK_MAGIC", "26092101")),
        )
