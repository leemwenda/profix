import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

from profx.webhook.app import create_app
from profx.webhook.executor import DryRunExecutor, MT5Executor, lots_for_risk
from profx.webhook.settings import Settings, SettingsError
from profx.webhook.store import SignalStore

SECRET = "unit-test-secret-0123456789abcdef"


class Spy(DryRunExecutor):
    def __init__(self):
        self.calls = []

    def execute(self, msg):
        self.calls.append(msg)
        return super().execute(msg)


def mk(tmp_path, clock=None, **kw):
    s = Settings(secret=SECRET, db_path=str(tmp_path / "s.db"), kill_file=str(tmp_path / "KILL"), **kw)
    spy = Spy()
    app = create_app(s, spy, SignalStore(s.db_path), clock or time.time)
    return TestClient(app), spy, s


def payload(**over):
    p = {"strategy_id": "PROFX_TREND_V1", "signal_id": f"EURUSD-{time.time_ns()}", "symbol": "EURUSD", "timeframe": "60",
         "action": "buy", "price": 1.1000, "sl": 1.0985, "tp": 1.1045, "risk_pct": 0.5, "timestamp": int(time.time()),
         "passphrase": SECRET}
    p.update(over)
    return {k: v for k, v in p.items() if v is not None}


def post(c, body, headers=None):
    raw = json.dumps(body).encode()
    return c.post("/webhook", content=raw, headers=headers or {})


def test_valid_signal_accepted_dry_run(tmp_path):
    c, spy, _ = mk(tmp_path)
    r = post(c, payload())
    assert r.status_code == 200 and r.json()["result"]["mode"] == "dry_run" and len(spy.calls) == 1


def test_wrong_or_missing_secret_rejected_before_anything_else(tmp_path):
    c, spy, _ = mk(tmp_path)
    assert post(c, payload(passphrase="wrong")).status_code == 401
    assert post(c, payload(passphrase=None)).status_code == 401
    assert not spy.calls


def test_hmac_signature_auth(tmp_path):
    c, spy, _ = mk(tmp_path)
    body = payload(passphrase=None)
    raw = json.dumps(body).encode()
    sig = hmac.new(SECRET.encode(), raw, hashlib.sha256).hexdigest()
    assert c.post("/webhook", content=raw, headers={"X-ProFX-Signature": "sha256=" + sig}).status_code == 200
    bad = c.post("/webhook", content=raw.replace(b"\"buy\"", b"\"sell\""), headers={"X-ProFX-Signature": "sha256=" + sig})
    assert bad.status_code == 401         # tampered body


def test_passphrase_can_be_disabled(tmp_path):
    c, _, _ = mk(tmp_path, allow_passphrase_auth=False)
    assert post(c, payload()).status_code == 401


def test_duplicate_signal_id_and_identical_body_replay_blocked(tmp_path):
    c, spy, _ = mk(tmp_path)
    p = payload(signal_id="dup-signal-0001")
    assert post(c, p).status_code == 200
    assert post(c, p).status_code == 409
    assert post(c, payload(signal_id="dup-signal-0001", price=1.1001)).status_code == 409   # same id, new body
    assert len(spy.calls) == 1


def test_replay_survives_restart(tmp_path):
    c, _, s = mk(tmp_path)
    p = payload(signal_id="persist-0001")
    assert post(c, p).status_code == 200
    c2 = TestClient(create_app(s, Spy(), SignalStore(s.db_path)))
    assert post(c2, p).status_code == 409


@pytest.mark.parametrize("delta", [-500, 500])
def test_stale_and_future_timestamps_rejected(tmp_path, delta):
    c, spy, _ = mk(tmp_path)
    assert post(c, payload(timestamp=int(time.time()) + delta)).status_code == 408 and not spy.calls


@pytest.mark.parametrize("over,code", [
    ({"symbol": "XAUUSD"}, 403), ({"strategy_id": "OTHER_STRAT"}, 403), ({"timeframe": "5"}, 422),
    ({"sl": None}, 422), ({"sl": 1.1010}, 422), ({"tp": 1.0990}, 422), ({"risk_pct": 2.0}, 422),
    ({"action": "hold"}, 422), ({"price": -1}, 422), ({"lots": 5.0}, 422), ({"signal_id": "x"}, 422),
])
def test_semantic_and_schema_rejections(tmp_path, over, code):
    c, spy, _ = mk(tmp_path)
    assert post(c, payload(**over)).status_code == code and not spy.calls


def test_sell_side_checks(tmp_path):
    c, _, _ = mk(tmp_path)
    assert post(c, payload(action="sell", price=1.1, sl=1.1015, tp=1.0955)).status_code == 200
    assert post(c, payload(action="sell", price=1.1, sl=1.0985)).status_code == 422


def test_kill_switch_and_rate_limit(tmp_path):
    c, spy, s = mk(tmp_path, max_signals_per_minute=2)
    assert post(c, payload()).status_code == 200 and post(c, payload()).status_code == 200
    assert post(c, payload()).status_code == 429
    (tmp_path / "KILL").write_text("x")
    assert post(c, payload()).status_code == 503
    assert len(spy.calls) == 2


def test_garbage_and_oversize_bodies(tmp_path):
    c, _, _ = mk(tmp_path)
    assert c.post("/webhook", content=b"not json").status_code == 400
    assert c.post("/webhook", content=b"[1,2]").status_code == 400
    assert c.post("/webhook", content=b"{" + b"a" * 5000).status_code == 413


def test_secret_never_echoed_or_logged(tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="profx.webhook")
    c, _, _ = mk(tmp_path)
    r = post(c, payload())
    assert SECRET not in r.text and SECRET not in caplog.text


def test_executor_crash_does_not_take_service_down(tmp_path):
    class Boom(DryRunExecutor):
        def execute(self, msg):
            raise RuntimeError("terminal gone")
    s = Settings(secret=SECRET, db_path=str(tmp_path / "s.db"), kill_file=str(tmp_path / "K"))
    c = TestClient(create_app(s, Boom(), SignalStore(s.db_path)))
    assert post(c, payload()).status_code == 502
    assert c.get("/health").status_code == 200


def test_settings_require_strong_secret(monkeypatch):
    monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
    with pytest.raises(SettingsError):
        Settings.from_env()
    monkeypatch.setenv("WEBHOOK_SECRET", "short")
    with pytest.raises(SettingsError):
        Settings.from_env()
    monkeypatch.setenv("WEBHOOK_SECRET", SECRET)
    assert Settings.from_env().dry_run is True     # safe default


def test_server_side_lot_sizing_rounds_down_and_skips_dust():
    # 10k equity, 0.5% = $50, 15 pip stop on EURUSD ($10/pip/lot) -> 0.33 lots
    assert lots_for_risk(10_000, 0.5, 0.0015, 0.00001, 1.0, 0.01, 100, 0.01) == pytest.approx(0.33)
    assert lots_for_risk(100, 0.5, 0.0015, 0.00001, 1.0, 0.01, 100, 0.01) == 0.0
    assert lots_for_risk(10_000, 0.5, 0.0, 0.00001, 1.0, 0.01, 100, 0.01) == 0.0


def test_mt5_executor_refuses_real_account_by_default():
    class FakeMT5:
        ACCOUNT_TRADE_MODE_DEMO = 0
        def initialize(self): return True
        def account_info(self):
            class A: trade_mode = 2
            return A()
    with pytest.raises(RuntimeError, match="REAL account"):
        MT5Executor(Settings(secret=SECRET), FakeMT5())
