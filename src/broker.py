"""Binance Spot adapter with multiple explicit safety gates.

The adapter is disabled unless the operator sets all required Render
variables. It supports manual approval and sandbox/testnet connectivity. It
never supports withdrawals. Spot mode accepts LONG plans only; SHORT plans
must use a futures adapter and are rejected here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import ccxt


class BrokerError(RuntimeError):
    pass


@dataclass(frozen=True)
class BrokerConfig:
    api_key: str = ""
    api_secret: str = ""
    sandbox: bool = True
    trading_enabled: bool = False
    mode: str = "manual"
    confirmation: str = ""

    @classmethod
    def from_env(cls) -> "BrokerConfig":
        return cls(
            api_key=os.getenv("BINANCE_API_KEY", ""),
            api_secret=os.getenv("BINANCE_API_SECRET", ""),
            sandbox=os.getenv("BINANCE_SANDBOX", "true").lower() == "true",
            trading_enabled=os.getenv("TRADING_ENABLED", "false").lower() == "true",
            mode=os.getenv("TRADING_MODE", "manual"),
            confirmation=os.getenv("LIVE_TRADING_CONFIRMATION", ""),
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    @property
    def submit_gate_open(self) -> bool:
        return (self.configured and self.trading_enabled and self.mode == "manual"
                and self.confirmation == "I UNDERSTAND")


class BinanceSpotBroker:
    def __init__(self, config: BrokerConfig | None = None, exchange=None):
        self.config = config or BrokerConfig.from_env()
        self.exchange = exchange
        if self.exchange is None and self.config.configured:
            self.exchange = ccxt.binance({
                "apiKey": self.config.api_key,
                "secret": self.config.api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            self.exchange.set_sandbox_mode(self.config.sandbox)

    def status(self) -> dict:
        return {
            "configured": self.config.configured,
            "sandbox": self.config.sandbox,
            "trading_enabled": self.config.trading_enabled,
            "mode": self.config.mode,
            "submit_gate_open": self.config.submit_gate_open,
            "withdrawals": False,
        }

    def _require_connection(self):
        if self.exchange is None:
            raise BrokerError("Binance API credentials are not configured.")

    def check_connection(self) -> dict:
        self._require_connection()
        balance = self.exchange.fetch_balance()
        usdt = balance.get("USDT", {})
        return {
            "connected": True,
            "sandbox": self.config.sandbox,
            "free_usdt": float(usdt.get("free", 0.0) or 0.0),
            "total_usdt": float(usdt.get("total", 0.0) or 0.0),
        }

    def fetch_order(self, order_id: str, symbol: str) -> dict:
        self._require_connection()
        return self.exchange.fetch_order(order_id, symbol)

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        self._require_connection()
        return self.exchange.cancel_order(order_id, symbol)

    def submit_long_bracket(self, symbol: str, quantity: float,
                            stop_loss: float, take_profit: float,
                            client_order_id: str):
        """Submit a spot market entry and protective exits after approval.

        Binance/ccxt order-type behavior can vary by account configuration, so
        this is gated and returns an error rather than leaving an unprotected
        position when either protective order cannot be created.
        """
        if not self.config.submit_gate_open:
            raise BrokerError("Manual live-order gate is closed.")
        if quantity <= 0 or stop_loss <= 0 or take_profit <= 0:
            raise BrokerError("Invalid quantity or protective levels.")
        self._require_connection()
        if not symbol.endswith("/USDT"):
            raise BrokerError("This first adapter only permits USDT spot pairs.")

        entry = None
        protection = []
        try:
            entry = self.exchange.create_order(
                symbol, "market", "buy", quantity,
                params={"newClientOrderId": client_order_id})
            filled = float(entry.get("filled") or quantity)
            # These are intentionally explicit limit-style protective orders;
            # the caller must reconcile and cancel the sibling after one fills.
            stop = self.exchange.create_order(
                symbol, "stop_loss_limit", "sell", filled, stop_loss,
                {"stopPrice": stop_loss, "timeInForce": "GTC"})
            protection.append(stop.get("id"))
            target = self.exchange.create_order(
                symbol, "take_profit_limit", "sell", filled, take_profit,
                {"stopPrice": take_profit, "timeInForce": "GTC"})
            protection.append(target.get("id"))
            return {"entry": entry, "protection": protection}
        except Exception as exc:
            # A failed protection placement is a hard failure. Do not claim a
            # protected position; the operator must reconcile immediately.
            raise BrokerError(f"Entry/protection sequence failed: {type(exc).__name__}") from exc
