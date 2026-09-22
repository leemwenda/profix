"""Executors. The client NEVER supplies lot size: lots are computed here from risk_pct and SL."""
from __future__ import annotations

import json
import logging
import math
import sys
from typing import Protocol

from .validation import SignalMessage

log = logging.getLogger("profx.webhook")


def lots_for_risk(equity: float, risk_pct: float, sl_distance: float, tick_size: float, tick_value_loss: float,
                  vol_min: float, vol_max: float, vol_step: float) -> float:
    """Lots so that a stop-out loses <= risk_pct of equity. Rounds DOWN; 0.0 means 'skip trade'."""
    if min(equity, risk_pct, sl_distance, tick_size, tick_value_loss, vol_step) <= 0:
        return 0.0
    loss_per_lot = sl_distance / tick_size * tick_value_loss
    raw = equity * risk_pct / 100.0 / loss_per_lot
    lots = math.floor(raw / vol_step + 1e-9) * vol_step
    lots = min(lots, vol_max)
    return round(lots, 8) if lots >= vol_min - 1e-12 else 0.0


class Executor(Protocol):
    def execute(self, msg: SignalMessage) -> dict: ...


class DryRunExecutor:
    """Default. Logs what WOULD happen; places nothing."""

    def execute(self, msg: SignalMessage) -> dict:
        out = {"mode": "dry_run", "would": msg.action, "symbol": msg.symbol, "sl": msg.sl, "tp": msg.tp, "risk_pct": msg.risk_pct}
        log.info(json.dumps({"event": "dry_run_order", "signal_id": msg.signal_id, **out}))
        return out


class MT5Executor:
    """Real execution through the official `MetaTrader5` package (Windows, terminal running).

    NOT exercised by the unit tests (needs a live terminal). Validate on a DEMO account first.
    """

    def __init__(self, settings, mt5_module=None):
        self.s = settings
        if mt5_module is None:
            import MetaTrader5 as mt5  # type: ignore
            mt5_module = mt5
        self.mt5 = mt5_module
        if not self.mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {self.mt5.last_error()}")
        acc = self.mt5.account_info()
        if acc is None:
            raise RuntimeError("no MT5 account info")
        if acc.trade_mode != self.mt5.ACCOUNT_TRADE_MODE_DEMO and not settings.allow_real_account:
            raise RuntimeError("Refusing to run on a REAL account (set WEBHOOK_ALLOW_REAL=1 deliberately).")

    def execute(self, msg: SignalMessage) -> dict:
        mt5, s = self.mt5, self.s
        if msg.action == "close":
            return self._close_all(msg)
        info = mt5.symbol_info(msg.symbol)
        if info is None or not mt5.symbol_select(msg.symbol, True):
            return {"status": "rejected", "reason": "symbol_unavailable"}
        if info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
            return {"status": "rejected", "reason": "market_closed_or_restricted"}
        tick = mt5.symbol_info_tick(msg.symbol)
        if tick is None:
            return {"status": "rejected", "reason": "no_tick"}
        pip = info.point * (10 if info.digits in (3, 5) else 1)
        if (tick.ask - tick.bid) / pip > s.max_spread_pips:
            return {"status": "rejected", "reason": "spread_too_wide"}
        buy = msg.action == "buy"
        px = tick.ask if buy else tick.bid
        if abs(px - msg.price) / msg.price * 100 > s.max_price_dev_pct:
            return {"status": "rejected", "reason": "price_deviation"}
        if mt5.positions_get(symbol=msg.symbol) and any(p.magic == s.magic for p in mt5.positions_get(symbol=msg.symbol)):
            return {"status": "rejected", "reason": "position_exists"}       # no duplicates / no accidental hedging
        sl_dist = abs(px - msg.sl)
        if sl_dist < info.trade_stops_level * info.point:
            return {"status": "rejected", "reason": "sl_inside_stop_level"}
        lots = lots_for_risk(mt5.account_info().equity, msg.risk_pct, sl_dist, info.trade_tick_size,
                             info.trade_tick_value_loss or info.trade_tick_value, info.volume_min, info.volume_max, info.volume_step)
        if lots <= 0:
            return {"status": "rejected", "reason": "lot_below_min"}
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": msg.symbol, "volume": lots,
               "type": mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL, "price": px, "sl": msg.sl,
               "tp": msg.tp or 0.0, "deviation": 10, "magic": s.magic, "comment": msg.signal_id[:30],
               "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC}
        chk = mt5.order_check(req)
        if chk is None or chk.retcode != 0:
            return {"status": "rejected", "reason": "order_check_failed", "retcode": getattr(chk, "retcode", None)}
        res = mt5.order_send(req)            # NO blind retry: a timeout could still have filled
        ok = res is not None and res.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL)
        out = {"status": "filled" if ok else "failed", "retcode": getattr(res, "retcode", None), "lots": lots,
               "ticket": getattr(res, "order", None), "price": getattr(res, "price", None)}
        log.info(json.dumps({"event": "order_result", "signal_id": msg.signal_id, **out}))
        return out

    def _close_all(self, msg: SignalMessage) -> dict:
        mt5, s = self.mt5, self.s
        closed = []
        for p in mt5.positions_get(symbol=msg.symbol) or []:
            if p.magic != s.magic:
                continue                                        # never touch manual / other-EA trades
            tick = mt5.symbol_info_tick(msg.symbol)
            buy = p.type == mt5.POSITION_TYPE_BUY
            r = mt5.order_send({"action": mt5.TRADE_ACTION_DEAL, "symbol": msg.symbol, "volume": p.volume, "position": p.ticket,
                                "type": mt5.ORDER_TYPE_SELL if buy else mt5.ORDER_TYPE_BUY,
                                "price": tick.bid if buy else tick.ask, "deviation": 10, "magic": s.magic,
                                "comment": "webhook_close", "type_filling": mt5.ORDER_FILLING_IOC})
            closed.append({"ticket": p.ticket, "retcode": getattr(r, "retcode", None)})
        return {"status": "closed", "positions": closed}
