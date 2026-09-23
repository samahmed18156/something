"""Automatic performance scorecard for completed manual/paper trades."""
from __future__ import annotations

from math import prod

import numpy as np
import pandas as pd


def _number(value, default=None):
    try:
        if value is None or value == "":
            return default
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _closed_frame(trades: list[dict]) -> pd.DataFrame:
    rows = []
    for trade in trades or []:
        if trade.get("status") == "OPEN" or trade.get("pnl_pct") is None:
            continue
        entry = _number(trade.get("entry"))
        stop = _number(trade.get("stop"))
        pnl = _number(trade.get("pnl_pct"))
        if entry is None or stop is None or pnl is None or entry <= 0:
            continue
        risk_pct = abs(entry - stop) / entry * 100.0
        rows.append({
            "id": trade.get("id", ""),
            "coin": trade.get("coin", "—"),
            "timeframe": trade.get("timeframe", "—"),
            "direction": trade.get("direction", "LONG"),
            "pnl_pct": pnl,
            "risk_pct": risk_pct,
            "r_multiple": pnl / risk_pct if risk_pct > 0 else np.nan,
            "quality_grade": trade.get("quality_grade", "Unknown") or "Unknown",
            "regime": trade.get("regime", "Unknown") or "Unknown",
            "score": _number(trade.get("score")),
            "agreement_pct": _number(trade.get("agreement_pct")),
            "slippage_bps": _number(trade.get("slippage_bps")),
            "opened_at": trade.get("opened_at", ""),
            "closed_at": trade.get("closed_at", ""),
            "exit_reason": trade.get("exit_reason", ""),
        })
    return pd.DataFrame(rows)


def _max_losing_streak(values: list[float]) -> int:
    current = 0
    longest = 0
    for value in values:
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _equity_stats(values: list[float]) -> tuple[float, float]:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in values:
        equity *= 1.0 + value / 100.0
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, (equity / peak - 1.0) * 100.0)
    return (equity - 1.0) * 100.0, max_drawdown


def _summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "trades": 0, "wins": 0, "losses": 0, "breakeven": 0,
            "win_rate_pct": 0.0, "total_return_pct": 0.0,
            "average_pnl_pct": 0.0, "median_pnl_pct": 0.0,
            "average_r": 0.0, "expectancy_r": 0.0,
            "profit_factor": 0.0, "max_drawdown_pct": 0.0,
            "max_losing_streak": 0, "average_slippage_bps": None,
        }
    pnl = frame["pnl_pct"].astype(float).tolist()
    r_values = frame["r_multiple"].dropna().astype(float)
    wins = sum(x > 0 for x in pnl)
    losses = sum(x < 0 for x in pnl)
    gains = sum(x for x in pnl if x > 0)
    loss_total = abs(sum(x for x in pnl if x < 0))
    total_return, drawdown = _equity_stats(pnl)
    return {
        "trades": len(pnl),
        "wins": wins,
        "losses": losses,
        "breakeven": len(pnl) - wins - losses,
        "win_rate_pct": wins / len(pnl) * 100.0,
        "total_return_pct": total_return,
        "average_pnl_pct": float(np.mean(pnl)),
        "median_pnl_pct": float(np.median(pnl)),
        "average_r": float(r_values.mean()) if not r_values.empty else 0.0,
        "expectancy_r": float(r_values.mean()) if not r_values.empty else 0.0,
        "profit_factor": gains / loss_total if loss_total > 0 else (float("inf") if gains else 0.0),
        "max_drawdown_pct": drawdown,
        "max_losing_streak": _max_losing_streak(pnl),
        "average_slippage_bps": (
            float(frame["slippage_bps"].dropna().mean())
            if not frame["slippage_bps"].dropna().empty else None
        ),
    }


def _grouped(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for name, group in frame.groupby(key, dropna=False):
        stats = _summary(group)
        rows.append({
            key.replace("_", " ").title(): name,
            "Trades": stats["trades"],
            "Win rate": f"{stats['win_rate_pct']:.1f}%",
            "Average P&L": f"{stats['average_pnl_pct']:+.2f}%",
            "Average R": f"{stats['average_r']:+.2f}R",
            "Expectancy": f"{stats['expectancy_r']:+.2f}R",
            "Max drawdown": f"{stats['max_drawdown_pct']:.2f}%",
        })
    return pd.DataFrame(rows).sort_values("Trades", ascending=False)


def build_scorecard(trades: list[dict]) -> dict:
    """Return metrics and breakdowns for completed records only."""
    frame = _closed_frame(trades)
    summary = _summary(frame)
    recent = frame.sort_values("closed_at", ascending=False).head(20) if not frame.empty else frame
    display_columns = [
        "coin", "timeframe", "direction", "quality_grade", "regime",
        "pnl_pct", "r_multiple", "slippage_bps", "opened_at", "closed_at",
        "exit_reason",
    ]
    recent_display = recent[[c for c in display_columns if c in recent.columns]].copy()
    if not recent_display.empty:
        recent_display = recent_display.rename(columns={
            "coin": "Coin", "timeframe": "Timeframe", "direction": "Direction",
            "quality_grade": "Grade", "regime": "Regime", "pnl_pct": "P&L %",
            "r_multiple": "R multiple", "slippage_bps": "Slippage bps",
            "opened_at": "Opened", "closed_at": "Closed", "exit_reason": "Exit reason",
        })
    return {
        "summary": summary,
        "by_coin": _grouped(frame, "coin"),
        "by_timeframe": _grouped(frame, "timeframe"),
        "by_grade": _grouped(frame, "quality_grade"),
        "by_regime": _grouped(frame, "regime"),
        "recent": recent_display,
        "raw": frame,
    }
