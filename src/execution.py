"""Execution guardrails and phone-alert helpers.

This module does not place live orders. It checks whether a qualified signal is
fresh and executable under current public order-book conditions, estimates a
market fill, sizes the position from a risk budget, and can send an opt-in
Telegram alert when the user explicitly requests one.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd

from .data import BINANCE_HOSTS, USER_AGENT

TIMEFRAME_DELTA = {
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


class ExecutionDataError(RuntimeError):
    """Raised when a public order-book snapshot cannot be obtained."""


def fetch_order_book(symbol: str, limit: int = 20, timeout: int = 8) -> dict:
    """Fetch a public spot order book from the first reachable Binance host."""
    pair = symbol.replace("/", "")
    errors: list[str] = []
    for host in BINANCE_HOSTS:
        url = f"https://{host}/api/v3/depth?symbol={pair}&limit={int(limit)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = json.loads(response.read().decode())
            bids = [(float(price), float(quantity)) for price, quantity in raw.get("bids", [])]
            asks = [(float(price), float(quantity)) for price, quantity in raw.get("asks", [])]
            if not bids or not asks:
                raise ValueError("empty order book")
            return {"bids": bids, "asks": asks, "source": host}
        except Exception as exc:
            errors.append(f"{host}: {type(exc).__name__}")
    raise ExecutionDataError("Public order book unavailable: " + " | ".join(errors))


def estimate_market_fill(levels: list[tuple[float, float]], quantity: float) -> dict:
    """Estimate VWAP fill and slippage from price/quantity levels.

    ``levels`` must be asks for a buy or bids for a sell. The result is
    deterministic and intentionally conservative when available depth is too
    small: it returns ``filled=False`` rather than pretending a fill exists.
    """
    remaining = float(quantity)
    notional = 0.0
    filled_qty = 0.0
    for price, level_qty in levels:
        take = min(remaining, float(level_qty))
        if take <= 0:
            continue
        notional += take * float(price)
        filled_qty += take
        remaining -= take
        if remaining <= 1e-12:
            break
    if filled_qty <= 0 or remaining > max(1e-12, quantity * 1e-6):
        return {"filled": False, "quantity": filled_qty, "remaining": max(0.0, remaining)}
    return {
        "filled": True,
        "quantity": filled_qty,
        "remaining": 0.0,
        "notional": notional,
        "average_price": notional / filled_qty,
    }


def _freshness(timestamp, timeframe: str, now=None, max_delay_minutes: int = 10) -> dict:
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    opened = pd.Timestamp(timestamp)
    if opened.tzinfo is None:
        opened = opened.tz_localize("UTC")
    else:
        opened = opened.tz_convert("UTC")
    candle_close = opened + TIMEFRAME_DELTA.get(timeframe, pd.Timedelta(hours=1))
    age_minutes = (now - candle_close).total_seconds() / 60.0
    fresh = 0.0 <= age_minutes <= float(max_delay_minutes)
    return {
        "fresh": fresh,
        "age_minutes": age_minutes,
        "candle_close": candle_close,
        "expires_at": candle_close + pd.Timedelta(minutes=max_delay_minutes),
    }


def execution_guard(symbol: str, direction: str, entry: float, tp: float,
                    sl: float, quality: dict, timestamp, timeframe: str,
                    book: dict | None = None, account_equity: float = 1000.0,
                    risk_pct: float = 1.0, max_position_pct: float = 25.0,
                    open_risk_pct: float = 0.0, daily_loss_pct: float = 0.0,
                    max_open_risk_pct: float = 3.0,
                    max_daily_loss_pct: float = 3.0,
                    max_spread_bps: float = 12.0,
                    max_slippage_bps: float = 20.0,
                    max_delay_minutes: int = 10,
                    fee_pct: float = 0.1,
                    slippage_pct: float = 0.05) -> dict:
    """Return a conservative execution decision without placing an order."""
    result = {
        "action": "WAIT",
        "symbol": symbol,
        "direction": direction,
        "fresh": False,
        "reasons": [],
        "source": None,
        "spread_bps": None,
        "slippage_bps": None,
        "mid_price": None,
        "estimated_fill": None,
        "quantity": None,
        "notional": None,
        "risk_amount": None,
        "reward_risk": None,
        "open_risk_pct": open_risk_pct,
        "daily_loss_pct": daily_loss_pct,
        "candle_close": None,
        "expires_at": None,
    }

    freshness = _freshness(timestamp, timeframe, max_delay_minutes=max_delay_minutes)
    result.update({
        "fresh": freshness["fresh"],
        "age_minutes": freshness["age_minutes"],
        "candle_close": freshness["candle_close"],
        "expires_at": freshness["expires_at"],
    })
    if not freshness["fresh"]:
        result["reasons"].append("Signal is outside the fresh execution window.")
    if quality.get("action") != "TRADE":
        result["reasons"].append("A-grade quality gate has not approved this signal.")
    if direction not in ("LONG", "SHORT"):
        result["reasons"].append("There is no directional execution plan.")
    if account_equity <= 0:
        result["reasons"].append("Reference account equity must be positive.")
    if open_risk_pct >= max_open_risk_pct:
        result["reasons"].append("Maximum open-risk budget has been reached.")
    if daily_loss_pct >= max_daily_loss_pct:
        result["reasons"].append("Daily loss cap has been reached.")

    risk_per_unit = abs(float(entry) - float(sl))
    reward_per_unit = abs(float(tp) - float(entry))
    if risk_per_unit <= 0 or reward_per_unit <= 0:
        result["reasons"].append("Entry, TP, and SL do not form a valid plan.")
        return result

    rr = reward_per_unit / risk_per_unit
    result["reward_risk"] = round(rr, 3)
    if rr < 1.0:
        result["reasons"].append("Reward/risk is below 1.0 after level calculation.")

    if book is None:
        try:
            book = fetch_order_book(symbol)
        except ExecutionDataError as exc:
            result["reasons"].append(str(exc))
            return result

    bids = book.get("bids", [])
    asks = book.get("asks", [])
    if not bids or not asks:
        result["reasons"].append("Order book has no usable bid/ask levels.")
        return result

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 10000.0 if mid else 99999.0
    result.update({
        "source": book.get("source", "public order book"),
        "mid_price": mid,
        "spread_bps": round(spread_bps, 2),
    })
    if spread_bps > max_spread_bps:
        result["reasons"].append(
            f"Spread is {spread_bps:.1f} bps, above {max_spread_bps:.1f} bps.")

    # Risk-based notional, capped by the execution allocation limit.
    risk_budget = float(account_equity) * float(risk_pct) / 100.0
    risk_fraction = risk_per_unit / float(entry)
    notional_by_risk = risk_budget / risk_fraction if risk_fraction > 0 else 0.0
    notional_cap = float(account_equity) * float(max_position_pct) / 100.0
    notional = min(notional_by_risk, notional_cap)
    quantity = notional / float(entry) if entry else 0.0
    result.update({
        "risk_amount": round(notional * risk_fraction, 2),
        "notional": round(notional, 2),
        "quantity": quantity,
    })

    levels = asks if direction == "LONG" else bids
    fill = estimate_market_fill(levels, quantity)
    if not fill.get("filled"):
        result["reasons"].append("Top-of-book depth is insufficient for the risk-sized order.")
        return result
    fill_price = float(fill["average_price"])
    slippage_bps = abs(fill_price - mid) / mid * 10000.0 if mid else 99999.0
    result.update({
        "estimated_fill": fill_price,
        "slippage_bps": round(slippage_bps, 2),
        "quantity": fill["quantity"],
    })
    if slippage_bps > max_slippage_bps:
        result["reasons"].append(
            f"Estimated fill slippage is {slippage_bps:.1f} bps, above {max_slippage_bps:.1f} bps.")

    # Use the estimated fill, not the chart close, for the final execution EV.
    actual_risk = abs(fill_price - float(sl))
    actual_reward = abs(float(tp) - fill_price)
    actual_rr = actual_reward / actual_risk if actual_risk > 0 else 0.0
    cost_r = ((2.0 * (fee_pct + slippage_pct)) / 100.0 * fill_price / actual_risk
              if actual_risk > 0 else 0.0)
    p = quality.get("probability")
    if p is not None:
        result["expected_value_r"] = round(float(p) * actual_rr - (1.0 - float(p)) - cost_r, 3)
        if result["expected_value_r"] <= 0:
            result["reasons"].append("Estimated fill makes expected value non-positive.")

    if not result["reasons"]:
        result["action"] = "EXECUTION READY"
    return result


def format_alert(result: dict, quality: dict) -> str:
    """Create a concise phone notification without credentials or secrets."""
    expiry = result.get("expires_at")
    expiry_text = expiry.strftime("%H:%M UTC") if expiry is not None else "—"
    return (
        f"{quality.get('grade', 'WAIT')}-grade {result.get('direction', 'WAIT')} "
        f"{result.get('symbol', '')}\n"
        f"Entry {result.get('estimated_fill', '—')} | TP/SL from workspace\n"
        f"EV {quality.get('expected_value_r', '—')}R | spread {result.get('spread_bps', '—')} bps\n"
        f"Execution window ends {expiry_text}"
    )


def send_telegram_alert(message: str, token: str | None = None,
                        chat_id: str | None = None, timeout: int = 10) -> dict:
    """Send an opt-in Telegram alert using Render environment variables."""
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return {"sent": False, "reason": "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID first."}
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode()
    try:
        req = urllib.request.Request(url, data=payload, method="POST",
                                     headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode())
        return {"sent": bool(data.get("ok")), "reason": "Telegram accepted the alert."}
    except Exception as exc:
        return {"sent": False, "reason": f"Telegram alert failed: {type(exc).__name__}."}
