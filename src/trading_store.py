"""Durable trade state for manual/testnet execution.

SQLite is suitable for local and testnet use. On Render, point
TRADING_DB_PATH at a persistent disk or replace this store with Postgres before
live trading. The store is intentionally append-oriented so restarts do not
lose the order lifecycle.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone


class TradeStore:
    def __init__(self, path: str | None = None):
        self.path = path or os.getenv("TRADING_DB_PATH", "trading_state.sqlite3")
        self._lock = threading.RLock()
        self._init()

    def _connect(self):
        con = sqlite3.connect(self.path, timeout=15)
        con.row_factory = sqlite3.Row
        return con

    def _init(self):
        with self._lock, self._connect() as con:
            con.executescript("""
            CREATE TABLE IF NOT EXISTS trade_plans (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry REAL NOT NULL,
                take_profit REAL NOT NULL,
                stop_loss REAL NOT NULL,
                quality_grade TEXT NOT NULL,
                quality_probability REAL,
                expected_value_r REAL,
                status TEXT NOT NULL,
                broker_order_id TEXT,
                protection_order_ids TEXT,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trade_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                event TEXT NOT NULL,
                details TEXT NOT NULL
            );
            """)

    def create_plan(self, payload: dict) -> str:
        plan_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as con:
            con.execute(
                """INSERT INTO trade_plans
                (id, created_at, symbol, timeframe, direction, entry,
                 take_profit, stop_loss, quality_grade, quality_probability,
                 expected_value_r, status, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (plan_id, now, payload["symbol"], payload["timeframe"],
                 payload["direction"], payload["entry"], payload["take_profit"],
                 payload["stop_loss"], payload.get("quality_grade", "WAIT"),
                 payload.get("quality_probability"), payload.get("expected_value_r"),
                 "PENDING_APPROVAL", json.dumps(payload, default=str)),
            )
            self._event_locked(con, plan_id, "PLAN_CREATED", payload)
        return plan_id

    def list_plans(self, status: str | None = None, limit: int = 50) -> list[dict]:
        query = "SELECT * FROM trade_plans"
        args: list = []
        if status:
            query += " WHERE status = ?"
            args.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        args.append(int(limit))
        with self._lock, self._connect() as con:
            rows = con.execute(query, args).fetchall()
        return [dict(row) for row in rows]

    def update_status(self, plan_id: str, status: str, details: dict | None = None,
                      broker_order_id: str | None = None,
                      protection_order_ids: list[str] | None = None):
        with self._lock, self._connect() as con:
            con.execute(
                "UPDATE trade_plans SET status = ?, broker_order_id = COALESCE(?, broker_order_id), protection_order_ids = COALESCE(?, protection_order_ids) WHERE id = ?",
                (status, broker_order_id,
                 json.dumps(protection_order_ids) if protection_order_ids is not None else None,
                 plan_id),
            )
            self._event_locked(con, plan_id, status, details or {})

    def record_event(self, plan_id: str, event: str, details: dict | None = None):
        with self._lock, self._connect() as con:
            self._event_locked(con, plan_id, event, details or {})

    @staticmethod
    def _event_locked(con, plan_id: str, event: str, details: dict):
        con.execute(
            "INSERT INTO trade_events(plan_id, created_at, event, details) VALUES (?, ?, ?, ?)",
            (plan_id, datetime.now(timezone.utc).isoformat(), event,
             json.dumps(details, default=str)),
        )
