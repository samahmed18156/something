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
from .backtest import _simulate, warmup_bars
from .config import IndicatorSettings, MTF_DEFAULT_BY_TF
from .patterns import pattern_flags
from .signals import parent_mtf_array

ENTRY_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
EXIT_GRID = (-0.35, -0.25, -0.15, -0.05, 0.0, 0.10)


def walk_forward(df: pd.DataFrame, cfg: IndicatorSettings | None = None,
                 parent_df: pd.DataFrame | None = None, parent_timeframe: str = "",
                 fee_pct: float = 0.1, n_folds: int = 4,
                 entry_grid=ENTRY_GRID, exit_grid=EXIT_GRID,
                 oos_frac: float = 0.30, use_mtf: bool | None = None,
                 use_patterns: bool = False) -> dict:
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
    mtf_on = (MTF_DEFAULT_BY_TF.get(parent_timeframe, True)
              if use_mtf is None else use_mtf)
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
                eq, _, _ = _simulate(df, ind, cfg, s, ins_e, et, xt,
                                     fee_pct, mtf_arr, pat_bear,
                                     cfg.pattern_lookback)
                ret = (eq.iloc[-1] - 1.0) * 100.0
                if ret > best_ret:
                    best_ret, best_params = ret, (et, xt)

        # --- trade only the unseen slice ---
        eq_oos, trades_oos, _ = _simulate(df, ind, cfg, ins_e, e,
                                          best_params[0], best_params[1],
                                          fee_pct, mtf_arr, pat_bear,
                                          cfg.pattern_lookback)
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

    return {
        "folds": pd.DataFrame(folds),
        "oos_total_pct": (oos_mult - 1.0) * 100.0,
        "buy_hold_pct": (df["close"].iloc[-1] / df["close"].iloc[warm] - 1.0) * 100.0,
    }


def sensitivity_grid(df: pd.DataFrame, cfg: IndicatorSettings | None = None,
                     parent_df: pd.DataFrame | None = None,
                     parent_timeframe: str = "", fee_pct: float = 0.1,
                     entry_grid=ENTRY_GRID, exit_grid=EXIT_GRID,
                     use_patterns: bool = False) -> pd.DataFrame:
    """
    Full-sample total return (%) for every (entry, exit) combination.

    A wide plateau of decent returns = robust. A single bright spike =
    overfit. Index = entry threshold, columns = exit threshold.
    """
    cfg = cfg or IndicatorSettings()
    ind = ta.compute_all(df, cfg)
    mtf_arr = (parent_mtf_array(df, parent_df, parent_timeframe)
               if parent_timeframe else None)
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
            eq, _, _ = _simulate(df, ind, cfg, warm, len(df), et, xt,
                                 fee_pct, mtf_arr, pat_bear, cfg.pattern_lookback)
            row.append(round((eq.iloc[-1] - 1.0) * 100.0, 1))
        rows.append(row)

    return pd.DataFrame(
        rows,
        index=[f"enter ≥ {et:+.2f}" for et in entry_grid],
        columns=[f"exit ≤ {xt:+.2f}" for xt in exit_grid],
    )
