"""
Walk-forward validation + parameter sensitivity.

Indicators and consensus thresholds are easy to overfit. The walk-forward
here does the honest thing:

  1. Split the sample into N consecutive folds.
  2. In each fold, pick the best (entry, exit) threshold pair on the first
     70% (in-sample) — the "optimization" window.
  3. Trade ONLY the unseen last 30% (out-of-sample) with that pair.
  4. Chain the out-of-sample results → the number you can actually believe.

If the out-of-sample curve is solid, the consensus logic has some real
structure in it. If it collapses, the in-sample shine was noise.

`sensitivity_grid` runs every threshold pair on the FULL sample to show
where returns are robust (a wide green plateau) vs fragile (a spike).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ta
from .backtest import _mtf_enabled, _simulate, warmup_bars
from .config import IndicatorSettings, MIN_CONFIRMATIONS
from .patterns import pattern_flags
from .signals import (parent_mtf_array, quality_filters, regime_mults,
                      row_votes, score_of)

ENTRY_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
EXIT_GRID = (-0.35, -0.25, -0.15, -0.05, 0.0, 0.10)


def walk_forward(df: pd.DataFrame, cfg: IndicatorSettings | None = None,
                 parent_df: pd.DataFrame | None = None, parent_timeframe: str = "",
                 fee_pct: float = 0.1, n_folds: int = 4,
                 entry_grid=ENTRY_GRID, exit_grid=EXIT_GRID,
                 oos_frac: float = 0.30, use_mtf: bool | None = None,
                 use_patterns: bool = False, timeframe: str = "",
                 slippage_pct: float = 0.05) -> dict:
    """
    Returns:
        folds          — DataFrame, one row per fold (window, chosen
                         thresholds, in-sample %, out-of-sample %, trades)
        oos_total_pct  — chained out-of-sample return across all folds
        buy_hold_pct   — buy & hold over the whole sample

    use_mtf=None → per-timeframe default (MTF_DEFAULT_BY_TF).
    """
    cfg = cfg or IndicatorSettings()
    ind = ta.compute_all(df, cfg)
    mtf_on = _mtf_enabled(timeframe, use_mtf)
    mtf_arr = (parent_mtf_array(df, parent_df, parent_timeframe)
               if (mtf_on and parent_timeframe) else None)
    pat_bear = (pattern_flags(df)["pat_bear"].to_numpy()
                if use_patterns else None)
    warm = warmup_bars(cfg)

    n = len(df) - warm
    if n < n_folds * 40:
        raise ValueError("Not enough candles for walk-forward — fetch more (limit=1000).")

    fold_len = n // n_folds
    folds: list[dict] = []
    oos_mult = 1.0

    for f in range(n_folds):
        s = warm + f * fold_len
        e = warm + (f + 1) * fold_len if f < n_folds - 1 else len(df)
        ins_e = s + int((e - s) * (1.0 - oos_frac))

        # --- pick the best thresholds on the in-sample window ---
        best_params = (0.25, -0.25)
        best_ret = -1e18
        for et in entry_grid:
            for xt in exit_grid:
                if xt >= et:
                    continue
                eq, _, _, _ = _simulate(df, ind, cfg, s, ins_e, et, xt,
                                     fee_pct, mtf_arr, pat_bear,
                                     cfg.pattern_lookback, slippage_pct)
                ret = (eq.iloc[-1] - 1.0) * 100.0
                if ret > best_ret:
                    best_ret, best_params = ret, (et, xt)

        # --- trade only the unseen slice ---
        eq_oos, trades_oos, _, _ = _simulate(df, ind, cfg, ins_e, e,
                                              best_params[0], best_params[1],
                                              fee_pct, mtf_arr, pat_bear,
                                              cfg.pattern_lookback, slippage_pct)
        oos_ret = (eq_oos.iloc[-1] - 1.0) * 100.0
        oos_mult *= (1.0 + oos_ret / 100.0)

        folds.append({
            "fold": f + 1,
            "window": f"{df.index[s]:%Y-%m-%d} → {df.index[e - 1]:%Y-%m-%d}",
            "entry_th": best_params[0],
            "exit_th": best_params[1],
            "in-sample %": round(best_ret, 2),
            "out-of-sample %": round(oos_ret, 2),
            "OOS trades": len(trades_oos),
        })

    fold_df = pd.DataFrame(folds)
    oos_values = fold_df["out-of-sample %"] if not fold_df.empty else pd.Series(dtype=float)
    return {
        "folds": fold_df,
        "oos_total_pct": (oos_mult - 1.0) * 100.0,
        "buy_hold_pct": (df["close"].iloc[-1] / df["close"].iloc[warm] - 1.0) * 100.0,
        "oos_positive_folds": int((oos_values > 0).sum()),
        "oos_fold_count": int(len(oos_values)),
        "oos_average_fold_pct": float(oos_values.mean()) if len(oos_values) else 0.0,
        "oos_worst_fold_pct": float(oos_values.min()) if len(oos_values) else 0.0,
        "slippage_pct_per_side": slippage_pct,
    }


def sensitivity_grid(df: pd.DataFrame, cfg: IndicatorSettings | None = None,
                     parent_df: pd.DataFrame | None = None,
                     parent_timeframe: str = "", fee_pct: float = 0.1,
                     entry_grid=ENTRY_GRID, exit_grid=EXIT_GRID,
                     use_patterns: bool = False, use_mtf: bool | None = None,
                     timeframe: str = "", slippage_pct: float = 0.05) -> pd.DataFrame:
    """
    Full-sample total return (%) for every (entry, exit) combination.

    A wide plateau of decent returns = robust. A single bright spike =
    overfit. Index = entry threshold, columns = exit threshold.
    """
    cfg = cfg or IndicatorSettings()
    ind = ta.compute_all(df, cfg)
    mtf_on = _mtf_enabled(timeframe, use_mtf)
    mtf_arr = (parent_mtf_array(df, parent_df, parent_timeframe)
               if (mtf_on and parent_timeframe) else None)
    pat_bear = (pattern_flags(df)["pat_bear"].to_numpy()
                if use_patterns else None)
    warm = warmup_bars(cfg)

    rows = []
    for et in entry_grid:
        row = []
        for xt in exit_grid:
            if xt >= et:
                row.append(float("nan"))
                continue
            eq, _, _, _ = _simulate(df, ind, cfg, warm, len(df), et, xt,
                                     fee_pct, mtf_arr, pat_bear,
                                     cfg.pattern_lookback, slippage_pct)
            row.append(round((eq.iloc[-1] - 1.0) * 100.0, 1))
        rows.append(row)

    return pd.DataFrame(
        rows,
        index=[f"enter ≥ {et:+.2f}" for et in entry_grid],
        columns=[f"exit ≤ {xt:+.2f}" for xt in exit_grid],
    )


def confidence_report(df: pd.DataFrame, cfg: IndicatorSettings | None = None,
                      parent_df: pd.DataFrame | None = None,
                      parent_timeframe: str = "", timeframe: str = "",
                      entry_th: float = 0.25, fee_pct: float = 0.1,
                      slippage_pct: float = 0.05,
                      horizons=(1, 3, 5), use_mtf: bool | None = None,
                      use_patterns: bool = False) -> dict:
    """Estimate historical confidence for the current signal configuration.

    For each completed candle, the same core score and quality gates are
    calculated using only information available at that candle. Future closes
    are used only to score the outcome after the signal was generated. This is
    a descriptive confidence panel, not a guarantee or a new optimization
    target.
    """
    cfg = cfg or IndicatorSettings()
    horizons = tuple(sorted({int(h) for h in horizons if int(h) > 0}))
    if not horizons:
        raise ValueError("At least one positive horizon is required.")

    ind = ta.compute_all(df, cfg)
    mtf_on = _mtf_enabled(timeframe, use_mtf)
    mtf_arr = (parent_mtf_array(df, parent_df, parent_timeframe)
               if (mtf_on and parent_timeframe) else None)
    pat_bear = (pattern_flags(df)["pat_bear"].to_numpy()
                if use_patterns else None)
    warm = warmup_bars(cfg)
    max_h = max(horizons)

    outcomes = {h: [] for h in horizons}
    direction_outcomes = {"positive": [], "negative": []}
    direction_outcomes_by_horizon = {
        str(h): {"positive": [], "negative": []} for h in horizons
    }
    current = {"status": "Stable", "score": 0.0, "filter_reasons": []}

    def classify(i: int) -> tuple[int, float, list[str]]:
        weights, _, _ = regime_mults(ind, i, cfg)
        votes = row_votes(ind, i, cfg)
        score = score_of(votes, weights)
        bullish = sum(v == 1 for v in votes.values())
        bearish = sum(v == -1 for v in votes.values())
        direction = (1 if score >= entry_th and bullish >= MIN_CONFIRMATIONS
                     else -1 if score <= -entry_th and bearish >= MIN_CONFIRMATIONS
                     else 0)
        reasons: list[str] = []
        quality_ok, quality_reasons = quality_filters(ind, i, cfg)
        if not quality_ok:
            reasons.extend(quality_reasons)
        if mtf_arr is not None and mtf_arr[i] != 0:
            if (direction == 1 and mtf_arr[i] == -1) or (direction == -1 and mtf_arr[i] == 1):
                reasons.append("secondary trend conflict")
        if (direction == 1 and pat_bear is not None
                and any(pat_bear[j] for j in range(max(3, i - cfg.pattern_lookback + 1), i + 1))):
            reasons.append("reversal event")
        if reasons:
            direction = 0
        status = ("Upward" if direction == 1 else
                  "Downward" if direction == -1 else "Stable")
        return direction, score, reasons

    last_i = len(df) - 1
    if last_i >= 0:
        direction, score, reasons = classify(last_i)
        current = {"status": ("Upward" if direction == 1 else
                               "Downward" if direction == -1 else "Stable"),
                   "score": round(float(score), 4),
                   "filter_reasons": reasons}

    end = len(df) - max_h
    for i in range(warm, max(warm, end)):
        direction, _, reasons = classify(i)
        if direction == 0:
            continue
        for h in horizons:
            if i + h >= len(df):
                continue
            raw = (float(df["close"].iloc[i + h]) / float(df["close"].iloc[i]) - 1.0) * 100.0
            net = direction * raw - 2.0 * (fee_pct + slippage_pct)
            outcomes[h].append(net)
            direction_key = "positive" if direction == 1 else "negative"
            direction_outcomes[direction_key].append(net)
            direction_outcomes_by_horizon[str(h)][direction_key].append(net)

    def summarize(values: list[float]) -> dict:
        if not values:
            return {"samples": 0, "wins": 0, "win_rate_pct": 0.0,
                    "average_return_pct": 0.0, "median_return_pct": 0.0,
                    "best_return_pct": 0.0, "worst_return_pct": 0.0,
                    "profit_factor": 0.0}
        arr = np.asarray(values, dtype=float)
        wins = arr[arr > 0]
        losses = arr[arr <= 0]
        gross_loss = abs(losses.sum())
        return {
            "samples": int(len(arr)),
            "wins": int(len(wins)),
            "win_rate_pct": float(100.0 * len(wins) / len(arr)),
            "average_return_pct": float(arr.mean()),
            "median_return_pct": float(np.median(arr)),
            "best_return_pct": float(arr.max()),
            "worst_return_pct": float(arr.min()),
            "profit_factor": (float("inf") if gross_loss == 0 else float(wins.sum() / gross_loss)),
        }

    return {
        "current": current,
        "horizons": {str(h): summarize(outcomes[h]) for h in horizons},
        "directions": {key: summarize(values) for key, values in direction_outcomes.items()},
        "direction_horizons": {
            str(h): {key: summarize(values) for key, values in by_direction.items()}
            for h, by_direction in direction_outcomes_by_horizon.items()
        },
        "fee_pct_per_side": fee_pct,
        "slippage_pct_per_side": slippage_pct,
        "filter_enabled": use_patterns,
    }
