"""
Long-only backtest of the consensus engine.

Walks forward over historical candles: go long when the weighted score
crosses above the entry threshold (and the higher timeframe does not point
down), exit when it crosses below the exit threshold. Fees on both sides.

The engine uses the SAME code as live — 12 indicator votes, regime-aware
weights (Choppiness Index) and the multi-timeframe filter — so the backtest
measures what the live signal would actually have done. Derivatives
(funding/OI) are live-only snapshots and therefore not part of the
backtest; they shift the score by at most 2 votes.

This is intentionally a simple research baseline, not a production
strategy — pair it with src/validation.py (walk-forward) before believing
any result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ta
from .config import IndicatorSettings, MTF_DEFAULT_BY_TF
from .patterns import pattern_flags
from .signals import parent_mtf_array, regime_mults, row_votes, score_of


def warmup_bars(cfg: IndicatorSettings) -> int:
    """Bars needed before the slowest indicator (EMA200) is meaningful."""
    return max(cfg.ema_trend_slow, cfg.bb_period, cfg.adx_period * 2) + 10


def _simulate(df: pd.DataFrame, ind: pd.DataFrame, cfg: IndicatorSettings,
              s: int, e: int, entry_th: float, exit_th: float, fee_pct: float,
              mtf_arr=None, pat_bear: np.ndarray | None = None,
              pattern_lookback: int = 1):
    """
    Core loop over bars [s, e). Returns (equity series, trade list,
    entries_blocked_by_pattern).
    """
    cash = 1.0
    pos = 0
    entry_px = 0.0
    entry_i = -1
    blocked = 0
    trades: list[dict] = []
    equity = pd.Series(1.0, index=df.index[s:e])

    for i in range(s, e):
        weights, _, _ = regime_mults(ind, i, cfg)
        score = score_of(row_votes(ind, i, cfg), weights)
        px = float(df["close"].iloc[i])
        k = i - s

        allow_entry = mtf_arr is None or mtf_arr[i] >= 0
        pattern_veto = False
        if pat_bear is not None and allow_entry and score >= entry_th:
            for j in range(max(3, i - pattern_lookback + 1), i + 1):
                if pat_bear[j]:
                    pattern_veto = True
                    break
        if pattern_veto:
            blocked += 1
        if pos == 0 and score >= entry_th and allow_entry and not pattern_veto:
            entry_px = px * (1 + fee_pct / 100.0)
            entry_i = i
            pos = 1
        elif pos == 1 and score <= exit_th:
            exit_px = px * (1 - fee_pct / 100.0)
            trades.append(_trade(df, entry_i, i, entry_px, exit_px))
            cash *= exit_px / entry_px
            pos = 0

        equity.iloc[k] = cash * (px / entry_px if pos == 1 else 1.0)

    if pos == 1:  # close the open position at the last price
        last = float(df["close"].iloc[e - 1])
        exit_px = last * (1 - fee_pct / 100.0)
        trades.append(_trade(df, entry_i, e - 1, entry_px, exit_px))
        cash *= exit_px / entry_px
        equity.iloc[-1] = cash

    return equity, trades, blocked


def run_backtest(df: pd.DataFrame, cfg: IndicatorSettings | None = None,
                 entry_th: float = 0.25, exit_th: float = -0.25,
                 fee_pct: float = 0.1, parent_df: pd.DataFrame | None = None,
                 parent_timeframe: str = "", use_mtf: bool | None = None,
                 use_patterns: bool = False) -> dict:
    """
    Backtest one parameter set. Pass parent_df (higher timeframe, unclosed
    candle dropped) + parent_timeframe to activate the MTF filter.
    use_mtf=None → per-timeframe default (MTF_DEFAULT_BY_TF).
    use_patterns → candlestick-pattern entry veto. Default OFF: the A/B
    walk-forward evidence (README) shows it only helps on some coins
    (BTC 4h) and hurts on others (DOGE 4h) — flip it per coin via the
    dashboard A/B tab if you want it.

    Returns: equity, buy_hold, trades (DataFrame), stats (dict).
    """
    cfg = cfg or IndicatorSettings()
    ind = ta.compute_all(df, cfg)
    warm = warmup_bars(cfg)
    if len(df) <= warm + 10:
        raise ValueError(
            "Not enough candles for a backtest — fetch more (e.g. limit=1000).")

    mtf_on = (MTF_DEFAULT_BY_TF.get(parent_timeframe, True)
              if use_mtf is None else use_mtf)
    mtf_arr = (parent_mtf_array(df, parent_df, parent_timeframe)
               if (mtf_on and parent_timeframe) else None)

    pat_bear = None
    if use_patterns:
        pat_bear = pattern_flags(df)["pat_bear"].to_numpy()

    equity, trades, blocked = _simulate(
        df, ind, cfg, warm, len(df),
        entry_th, exit_th, fee_pct, mtf_arr, pat_bear, cfg.pattern_lookback)
    buy_hold = df["close"].iloc[warm:].copy()
    buy_hold = buy_hold / buy_hold.iloc[0]

    stats = _stats(equity, buy_hold, pd.DataFrame(trades))
    stats["entries_blocked_by_pattern"] = blocked
    return {
        "equity": equity,
        "buy_hold": buy_hold,
        "trades": pd.DataFrame(trades),
        "stats": stats,
    }


def _trade(df: pd.DataFrame, entry_i: int, exit_i: int,
           entry_px: float, exit_px: float) -> dict:
    pnl = (exit_px / entry_px - 1.0) * 100.0
    return {
        "entry_time": df.index[entry_i],
        "exit_time": df.index[exit_i],
        "entry": round(entry_px, 4),
        "exit": round(exit_px, 4),
        "pnl_pct": round(pnl, 2),
        "bars_held": exit_i - entry_i,
    }


def _stats(equity: pd.Series, buy_hold: pd.Series, t: pd.DataFrame) -> dict:
    total_return = (equity.iloc[-1] - 1.0) * 100.0
    bh = (buy_hold.iloc[-1] - 1.0) * 100.0
    max_dd = (equity / equity.cummax() - 1.0).min() * 100.0

    n = len(t)
    if n:
        wins = t[t["pnl_pct"] > 0]
        losses = t[t["pnl_pct"] <= 0]
        win_rate = 100.0 * len(wins) / n
        gross_win = wins["pnl_pct"].sum()
        gross_loss = abs(losses["pnl_pct"].sum())
        profit_factor = float("inf") if gross_loss == 0 else gross_win / gross_loss
        avg_trade = t["pnl_pct"].mean()
    else:
        win_rate, profit_factor, avg_trade = 0.0, 0.0, 0.0

    return {
        "total_return_pct": total_return,
        "buy_hold_pct": bh,
        "max_drawdown_pct": max_dd,
        "num_trades": n,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_pct": avg_trade,
    }
