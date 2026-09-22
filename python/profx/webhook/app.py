"""FastAPI app: POST /webhook. Pipeline order matters (cheap/safe checks first, side effects last)."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import deque

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .executor import DryRunExecutor, Executor
from .settings import Settings
from .store import SignalStore
from .validation import Rejected, authenticate, parse_body, validate_message

log = logging.getLogger("profx.webhook")


def create_app(settings: Settings, executor: Executor | None = None, store: SignalStore | None = None,
               clock=time.time) -> FastAPI:
    app = FastAPI(title="ProFX webhook", docs_url=None, redoc_url=None, openapi_url=None)  # no public API docs
    store = store or SignalStore(settings.db_path)
    executor = executor or DryRunExecutor()
    recent: deque[float] = deque()

    def audit(event: str, **kw) -> None:
        log.info(json.dumps({"ts": clock(), "event": event, **kw}, default=str))

    @app.get("/health")
    def health():
        return {"ok": True, "dry_run": settings.dry_run, "kill_switch": os.path.exists(settings.kill_file)}

    @app.post("/webhook")
    async def webhook(request: Request):
        raw = await request.body()
        now = clock()
        try:
            obj = parse_body(raw, settings)
            how = authenticate(raw, request.headers.get("x-profx-signature"), obj.get("passphrase"), settings)
            obj.pop("passphrase", None)                   # never keep/log the secret
            msg = validate_message(obj, settings, now)
            if os.path.exists(settings.kill_file):
                raise Rejected(503, "kill_switch_active")
            while recent and now - recent[0] > 60:
                recent.popleft()
            if len(recent) >= settings.max_signals_per_minute:
                raise Rejected(429, "rate_limited")
            body_hash = hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
            if not store.register(msg.signal_id, body_hash, now):
                raise Rejected(409, "duplicate_or_replay")
            recent.append(now)
            store.purge(now - settings.retention_days * 86400)
            try:
                result = executor.execute(msg)
            except Exception as exc:                      # executor failure must never crash the service
                store.set_status(msg.signal_id, "executor_error")
                audit("executor_error", signal_id=msg.signal_id, error=repr(exc))
                return JSONResponse({"status": "error", "code": "executor_error"}, status_code=502)
            store.set_status(msg.signal_id, str(result.get("status", result.get("mode", "done"))))
            audit("accepted", signal_id=msg.signal_id, symbol=msg.symbol, action=msg.action, auth=how, result=result)
            return {"status": "accepted", "signal_id": msg.signal_id, "result": result}
        except Rejected as r:
            audit("rejected", code=r.code, detail=r.detail, client=request.client.host if request.client else None)
            return JSONResponse({"status": "rejected", "code": r.code}, status_code=r.status)   # no detail leak

    return app
