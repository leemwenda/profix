"""Pure validation logic (no I/O) so it is unit-testable."""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from .settings import Settings


class Rejected(Exception):
    def __init__(self, status: int, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}")
        self.status, self.code, self.detail = status, code, detail


class SignalMessage(BaseModel):
    model_config = {"extra": "forbid"}          # unknown fields are rejected, not ignored
    version: int = 1
    strategy_id: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_\-]+$")
    signal_id: str = Field(min_length=8, max_length=80, pattern=r"^[A-Za-z0-9_\-:\.]+$")
    symbol: str = Field(min_length=6, max_length=12)
    timeframe: str = Field(max_length=6)
    action: Literal["buy", "sell", "close"]
    price: float = Field(gt=0)
    sl: float | None = Field(default=None, gt=0)
    tp: float | None = Field(default=None, gt=0)
    risk_pct: float = Field(gt=0, le=100)
    timestamp: int = Field(gt=1_500_000_000)     # epoch seconds (UTC) at signal creation
    passphrase: str | None = None

    @field_validator("symbol")
    @classmethod
    def _sym(cls, v: str) -> str:
        v = v.upper().replace("/", "")
        if not v.isalpha():
            raise ValueError("symbol must be letters only")
        return v


def verify_signature(raw: bytes, header: str | None, secret: str) -> bool:
    if not header:
        return False
    given = header.split("=", 1)[1] if header.startswith("sha256=") else header
    good = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(given.strip().lower(), good)


def authenticate(raw: bytes, sig_header: str | None, body_passphrase: str | None, s: Settings) -> str:
    if sig_header and verify_signature(raw, sig_header, s.secret):
        return "hmac"
    if s.allow_passphrase_auth and body_passphrase and hmac.compare_digest(body_passphrase.encode(), s.secret.encode()):
        return "passphrase"
    raise Rejected(401, "auth_failed")


def parse_body(raw: bytes, s: Settings) -> dict:
    if len(raw) > s.max_body_bytes:
        raise Rejected(413, "body_too_large")
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError
    except ValueError:
        raise Rejected(400, "invalid_json")
    return obj


def validate_message(obj: dict, s: Settings, now: float | None = None) -> SignalMessage:
    now = time.time() if now is None else now
    try:
        msg = SignalMessage(**obj)
    except ValidationError as e:
        raise Rejected(422, "schema", str(e.errors()[0]["msg"]) if e.errors() else "invalid")
    if abs(now - msg.timestamp) > s.ttl_seconds:
        raise Rejected(408, "stale_or_future_timestamp", f"age={now - msg.timestamp:.0f}s ttl={s.ttl_seconds}s")
    if msg.strategy_id not in s.allowed_strategies:
        raise Rejected(403, "strategy_not_allowed")
    if msg.symbol not in s.allowed_symbols:
        raise Rejected(403, "symbol_not_allowed")
    if msg.timeframe not in s.allowed_timeframes:
        raise Rejected(422, "timeframe_not_allowed")
    if msg.action in ("buy", "sell"):
        if msg.sl is None:
            raise Rejected(422, "sl_required")           # never open a position without a stop
        buy = msg.action == "buy"
        if (buy and not msg.sl < msg.price) or (not buy and not msg.sl > msg.price):
            raise Rejected(422, "sl_wrong_side")
        if msg.tp is not None and ((buy and not msg.tp > msg.price) or (not buy and not msg.tp < msg.price)):
            raise Rejected(422, "tp_wrong_side")
        if msg.risk_pct > s.max_risk_pct:
            raise Rejected(422, "risk_above_ceiling", f"{msg.risk_pct} > {s.max_risk_pct}")
    return msg
