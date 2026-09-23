"""Public order-book liquidity-wall analysis for manual signal confirmation.

This module deliberately does not identify a person or institution behind an
order and does not place orders. A single order-book snapshot is not enough to
call a wall genuine, so the analysis rewards persistence and executed trades
near the level when repeated on-demand scans are available.
"""
from __future__ import annotations

import json
import statistics
import time
import urllib.request

from .data import BINANCE_HOSTS, USER_AGENT
from .execution import fetch_order_book


class OrderFlowDataError(RuntimeError):
    """Raised when public recent-trade data cannot be obtained."""


def fetch_recent_trades(symbol: str, limit: int = 100, timeout: int = 8) -> list[dict]:
    """Fetch public recent spot trades from Binance, without API credentials."""
    pair = symbol.replace("/", "")
    errors: list[str] = []
    for host in BINANCE_HOSTS:
        url = f"https://{host}/api/v3/trades?symbol={pair}&limit={int(limit)}"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.loads(response.read().decode())
            return [
                {
                    "price": float(row["price"]),
                    "quantity": float(row["qty"]),
                    # Binance's buyer-maker flag means the taker was a seller.
                    "buyer_maker": bool(row.get("isBuyerMaker", False)),
                    "time": row.get("time"),
                }
                for row in raw
            ]
        except Exception as exc:
            errors.append(f"{host}: {type(exc).__name__}")
    raise OrderFlowDataError("Recent public trades unavailable: " + " | ".join(errors))


def _tick_size(levels: list[tuple[float, float]]) -> float:
    prices = sorted({float(price) for price, _ in levels if float(price) > 0})
    differences = [b - a for a, b in zip(prices, prices[1:]) if b > a]
    return min(differences) if differences else (prices[0] * 1e-6 if prices else 0.0)


def _wall_candidate(levels: list[tuple[float, float]], mid: float,
                    side: str, max_distance_bps: float) -> dict | None:
    if not levels or mid <= 0:
        return None
    notionals = [float(price) * float(quantity)
                 for price, quantity in levels
                 if float(price) > 0 and float(quantity) > 0]
    if not notionals:
        return None
    median_notional = statistics.median(notionals)
    if median_notional <= 0:
        return None

    candidates = []
    for price, quantity in levels:
        price, quantity = float(price), float(quantity)
        if price <= 0 or quantity <= 0:
            continue
        distance_bps = ((mid - price) if side == "BID" else (price - mid)) / mid * 10000.0
        if distance_bps < 0 or distance_bps > max_distance_bps:
            continue
        notional = price * quantity
        candidates.append({
            "side": side,
            "price": price,
            "quantity": quantity,
            "notional": notional,
            "distance_bps": distance_bps,
            "size_multiple": notional / median_notional,
        })
    if not candidates:
        return None
    strongest = max(candidates, key=lambda row: row["size_multiple"])
    # A level that is only about twice the nearby median is ordinary depth,
    # not a useful liquidity-wall candidate. Keep the detector conservative.
    return strongest if strongest["size_multiple"] >= 3.0 else None


def _trade_flow(trades: list[dict] | None, wall: dict | None) -> dict:
    buy_notional = 0.0
    sell_notional = 0.0
    near_wall_sell = 0.0
    near_wall_buy = 0.0
    tolerance = max(3.0, float(wall["distance_bps"]) * 0.15) if wall else 3.0
    wall_price = float(wall["price"]) if wall else None

    for trade in trades or []:
        try:
            price = float(trade["price"])
            notional = price * float(trade["quantity"])
        except (KeyError, TypeError, ValueError):
            continue
        if notional <= 0:
            continue
        buyer_maker = bool(trade.get("buyer_maker", trade.get("isBuyerMaker", False)))
        if buyer_maker:
            sell_notional += notional
        else:
            buy_notional += notional
        if wall_price and wall_price > 0:
            near_bps = abs(price - wall_price) / wall_price * 10000.0
            if near_bps <= tolerance:
                if buyer_maker:
                    near_wall_sell += notional
                else:
                    near_wall_buy += notional

    total = buy_notional + sell_notional
    return {
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "flow_imbalance": (buy_notional - sell_notional) / total if total else 0.0,
        "near_wall_sell_notional": near_wall_sell,
        "near_wall_buy_notional": near_wall_buy,
        "trades_count": len(trades or []),
    }


def _same_wall(previous: dict | None, wall: dict | None) -> bool:
    if not previous or not wall:
        return False
    if previous.get("side") != wall.get("side"):
        return False
    previous_price = float(previous.get("price", 0.0))
    current_price = float(wall.get("price", 0.0))
    if previous_price <= 0 or current_price <= 0:
        return False
    tolerance_bps = max(2.0, float(previous.get("tick_size", 0.0)) / previous_price * 10000.0 * 2.0)
    return abs(current_price - previous_price) / previous_price * 10000.0 <= tolerance_bps


def analyze_liquidity(book: dict, trades: list[dict] | None = None,
                     previous: dict | None = None, now: float | None = None,
                     max_distance_bps: float = 100.0) -> dict:
    """Analyze a public snapshot and return a manual-only liquidity assessment.

    ``previous`` is the tracking object returned in ``result['tracking']`` by
    an earlier scan. Keeping the same wall across scans is the only way this
    on-demand mode can estimate persistence; one snapshot is labelled as such.
    """
    now = time.time() if now is None else float(now)
    bids = [(float(price), float(quantity)) for price, quantity in book.get("bids", [])]
    asks = [(float(price), float(quantity)) for price, quantity in book.get("asks", [])]
    if not bids or not asks:
        raise OrderFlowDataError("Order book has no usable bid and ask levels.")

    best_bid, best_ask = bids[0][0], asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 10000.0 if mid else 99999.0
    levels = bids + asks
    tick_size = _tick_size(levels)
    bid_wall = _wall_candidate(bids[:50], mid, "BID", max_distance_bps)
    ask_wall = _wall_candidate(asks[:50], mid, "ASK", max_distance_bps)
    candidates = [wall for wall in (bid_wall, ask_wall) if wall is not None]
    wall = max(candidates, key=lambda row: row["size_multiple"]) if candidates else None

    flow = _trade_flow(trades, wall)
    same_wall = _same_wall(previous, wall)
    wall_age = 0.0
    size_change_pct = None
    cancel_risk = "MEDIUM"
    if same_wall and previous:
        first_seen = float(previous.get("first_seen", now))
        wall_age = max(0.0, now - first_seen)
        previous_notional = float(previous.get("notional", 0.0))
        if previous_notional > 0:
            size_change_pct = (float(wall["notional"]) / previous_notional - 1.0) * 100.0
        if size_change_pct is not None and size_change_pct < -60.0:
            cancel_risk = "HIGH"
        elif wall_age >= 60.0 and (size_change_pct is None or size_change_pct > -40.0):
            cancel_risk = "LOW"
    elif previous and wall is None:
        cancel_risk = "HIGH"

    action = "NO WALL"
    score = 0.0
    reasons: list[str] = []
    proposed_entry = None
    if wall is not None:
        size_score = min(1.0, max(0.0, (wall["size_multiple"] - 2.0) / 6.0))
        distance_score = max(0.0, 1.0 - wall["distance_bps"] / max_distance_bps)
        persistence_score = min(1.0, wall_age / 60.0)
        if wall["side"] == "BID":
            absorption_score = min(1.0, flow["near_wall_sell_notional"] / max(wall["notional"] * 0.25, 1e-9))
        else:
            absorption_score = min(1.0, flow["near_wall_buy_notional"] / max(wall["notional"] * 0.25, 1e-9))
        spread_score = max(0.0, 1.0 - spread_bps / 20.0)
        score = round(100.0 * (
            0.35 * size_score + 0.20 * distance_score +
            0.20 * absorption_score + 0.15 * persistence_score +
            0.10 * spread_score), 1)

        proposed_entry = wall["price"] + tick_size if wall["side"] == "BID" else wall["price"] - tick_size
        if wall_age < 20.0:
            reasons.append("Persistence is not established; scan again after 20–30 seconds.")
        if wall["side"] == "BID" and flow["near_wall_sell_notional"] <= 0:
            reasons.append("No recent seller aggression was detected at the bid wall.")
        if wall["side"] == "ASK":
            reasons.append("The strongest wall is an ask wall; this is not a Spot LONG confirmation.")
        if spread_bps > 12.0:
            reasons.append(f"Spread is {spread_bps:.1f} bps, which is wide for a precise entry.")

        if (wall["side"] == "BID" and score >= 70.0 and wall_age >= 20.0
                and flow["near_wall_sell_notional"] > 0 and spread_bps <= 12.0
                and cancel_risk != "HIGH"):
            action = "CONFIRMED BID WALL"
        else:
            action = "WATCH"
    else:
        reasons.append("No unusually large visible wall was found within the scan range.")
        if previous:
            reasons.append("The previously tracked wall is no longer visible; treat that as cancellation risk.")

    tracking = None
    if wall is not None:
        tracking = {
            "side": wall["side"],
            "price": wall["price"],
            "notional": wall["notional"],
            "tick_size": tick_size,
            "first_seen": (float(previous.get("first_seen", now))
                           if same_wall and previous else now),
        }

    return {
        "action": action,
        "score": score,
        "symbol": book.get("symbol"),
        "source": book.get("source", "public order book"),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid,
        "spread_bps": spread_bps,
        "wall": wall,
        "wall_age_seconds": wall_age,
        "size_change_pct": size_change_pct,
        "cancel_risk": cancel_risk,
        "proposed_entry": proposed_entry,
        "flow": flow,
        "reasons": reasons,
        "tracking": tracking,
        "scanned_at": now,
    }


def scan_liquidity(symbol: str, previous: dict | None = None,
                   depth_limit: int = 50, trade_limit: int = 100) -> dict:
    """Fetch current public depth and trades, then analyze them."""
    book = fetch_order_book(symbol, limit=depth_limit)
    book["symbol"] = symbol
    trades = fetch_recent_trades(symbol, limit=trade_limit)
    return analyze_liquidity(book, trades=trades, previous=previous)
