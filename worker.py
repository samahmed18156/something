"""Render background-worker entry point for manual/testnet execution.

The worker is intentionally disabled unless WORKER_ENABLED=true. In manual
mode it never discovers or submits a trade by itself. It reconciles already
approved/protected plans and records when one protective order closes so the
sibling can be cancelled.
"""
from __future__ import annotations

import json
import os
import time

from src.broker import BinanceSpotBroker
from src.trading_store import TradeStore


def reconcile_protected_plan(row: dict, store: TradeStore,
                             broker: BinanceSpotBroker) -> dict:
    ids = json.loads(row.get("protection_order_ids") or "[]")
    statuses = []
    for order_id in ids:
        if not order_id:
            continue
        try:
            order = broker.fetch_order(order_id, row["symbol"])
            statuses.append((order_id, order.get("status", "unknown")))
        except Exception as exc:
            store.record_event(row["id"], "RECONCILIATION_ERROR",
                               {"order_id": order_id, "error": type(exc).__name__})
            return {"status": "RECONCILIATION_ERROR", "orders": statuses}

    closed = [order_id for order_id, status in statuses if status == "closed"]
    if closed:
        for order_id, status in statuses:
            if status in {"open", "new", "partially_filled"} and order_id not in closed:
                try:
                    broker.cancel_order(order_id, row["symbol"])
                except Exception as exc:
                    store.record_event(row["id"], "SIBLING_CANCEL_ERROR",
                                       {"order_id": order_id, "error": type(exc).__name__})
        store.update_status(row["id"], "CLOSED",
                            {"closed_order_ids": closed, "statuses": statuses})
        return {"status": "CLOSED", "orders": statuses}
    return {"status": "PROTECTED", "orders": statuses}


def main():
    if os.getenv("WORKER_ENABLED", "false").lower() != "true":
        print("Worker disabled. Set WORKER_ENABLED=true only after testnet setup.")
        return
    store = TradeStore()
    broker = BinanceSpotBroker()
    print("Manual-approval trading worker started.")
    while True:
        protected = store.list_plans(status="PROTECTED")
        for row in protected:
            reconcile_protected_plan(row, store, broker)
        pending = len(store.list_plans(status="PENDING_APPROVAL"))
        print(f"protected plans: {len(protected)}; pending approval: {pending}; broker: {broker.status()}")
        time.sleep(int(os.getenv("WORKER_INTERVAL_SECONDS", "60")))


if __name__ == "__main__":
    main()
