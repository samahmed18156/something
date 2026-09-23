"""Manual approval orchestration for Binance Spot.

This module is the only place that can move a stored plan toward an exchange
order. It requires a stored pending plan, the manual approval flag, a LONG
plan (spot cannot short), and a broker whose explicit gate is open.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .broker import BinanceSpotBroker, BrokerError
from .trading_store import TradeStore


class LiveTradingError(RuntimeError):
    pass


def make_plan(symbol: str, timeframe: str, direction: str, entry: float,
              take_profit: float, stop_loss: float, quality: dict,
              execution: dict) -> dict:
    if direction != "LONG":
        raise LiveTradingError("Binance Spot manual execution only supports LONG plans.")
    if execution.get("action") != "EXECUTION READY":
        raise LiveTradingError("Execution guard has not approved the plan.")
    if quality.get("action") != "TRADE":
        raise LiveTradingError("A-grade quality gate has not approved the plan.")
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "entry": float(entry),
        "take_profit": float(take_profit),
        "stop_loss": float(stop_loss),
        "quantity": float(execution["quantity"]),
        "quality_grade": quality.get("grade", "WAIT"),
        "quality_probability": quality.get("probability"),
        "expected_value_r": quality.get("expected_value_r"),
        "execution": execution,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def approve_and_submit(plan_id: str, store: TradeStore,
                       broker: BinanceSpotBroker) -> dict:
    plans = [p for p in store.list_plans(status="PENDING_APPROVAL") if p["id"] == plan_id]
    if not plans:
        raise LiveTradingError("Plan is missing or is no longer pending approval.")
    row = plans[0]
    import json
    payload = json.loads(row["payload"])
    store.update_status(plan_id, "APPROVAL_CONFIRMED", {"operator": "manual"})
    try:
        result = broker.submit_long_bracket(
            row["symbol"], float(payload["quantity"]), row["stop_loss"],
            row["take_profit"], f"plan_{plan_id[:20]}")
        protection_ids = result.get("protection", [])
        entry_id = (result.get("entry") or {}).get("id")
        store.update_status(plan_id, "PROTECTED", {
            "entry_order_id": entry_id,
            "protection_order_ids": protection_ids,
        }, broker_order_id=entry_id, protection_order_ids=protection_ids)
        return {"status": "PROTECTED", "entry": result.get("entry"), "protection": protection_ids}
    except BrokerError as exc:
        store.update_status(plan_id, "PROTECTION_FAILED", {"error": str(exc)})
        raise
