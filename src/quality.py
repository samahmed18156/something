"""
A-grade signal qualification for the 1h workflow.

This module is deliberately conservative. The normal consensus engine creates
an initial directional signal; this module acts as a meta-label and decides
whether that signal is historically qualified for action.

Historical outcomes use a triple barrier label:
- target touched first within the horizon -> win
- stop touched first, or neither barrier touched -> non-win
- if both barriers are touched in one candle, the conservative stop-first
  assumption is used because intrabar order is unknown

The current signal is never included in its own historical sample. Probabilities
are Laplace-smoothed and accompanied by the sample count. They are estimates,
not guarantees.
"""
from __future__ import annotations

from datetime import timedelta
from math import sqrt

import numpy as np
import pandas as pd

from . import indicators as ta
from .backtest import _mtf_enabled, warmup_bars
from .config import IndicatorSettings, MIN_CONFIRMATIONS
from .patterns import pattern_flags
from .signals import parent_mtf_array, quality_filters, regime_mults, row_votes, score_of

QUALITY_HORIZON_BY_TF = {
    "15m": 4,
    "1h": 3,
    "4h": 3,
    "1d": 2,
}
MIN_HISTORICAL_SAMPLES = 20
MIN_PROBABILITY = 0.55
MIN_AGREEMENT = 55
MIN_SCORE = 0.25


def _barrier_label(df: pd.DataFrame, start: int, direction: int,
                   target: float, stop: float, horizon: int) -> int:
    """Return 1 only when target is reached before stop; otherwise 0."""
    end = min(len(df), start + horizon + 1)
    for i in range(start + 1, end):
        high = float(df["high"].iloc[i])
        low = float(df["low"].iloc[i])
        if direction == 1:
            hit_target = high >= target
            hit_stop = low <= stop
        else:
            hit_target = low <= target
            hit_stop = high >= stop

        # Intrabar order is unknowable. Stop-first avoids optimistic labels.
        if hit_stop:
            return 0
        if hit_target:
            return 1
    return 0


def _laplace_probability(wins: int, samples: int) -> float:
    if samples <= 0:
        return 0.5
    return (wins + 1.0) / (samples + 2.0)


def _wilson_lower(wins: int, samples: int, z: float = 1.96) -> float:
    """95% lower confidence bound for a Bernoulli win rate."""
    if samples <= 0:
        return 0.0
    p = wins / samples
    denominator = 1.0 + z * z / samples
    centre = p + z * z / (2.0 * samples)
    spread = z * sqrt((p * (1.0 - p) + z * z / (4.0 * samples)) / samples)
    return max(0.0, (centre - spread) / denominator)


def _timeframe_delta(timeframe: str) -> pd.Timedelta:
    return {
        "15m": pd.Timedelta(minutes=15),
        "1h": pd.Timedelta(hours=1),
        "4h": pd.Timedelta(hours=4),
        "1d": pd.Timedelta(days=1),
    }.get(timeframe, pd.Timedelta(hours=1))


def _historical_direction(df: pd.DataFrame, ind: pd.DataFrame, i: int,
                          cfg: IndicatorSettings, mtf_arr, pat_bear,
                          entry_threshold: float = 0.25) -> int:
    """Recreate the live direction using information available at candle i."""
    weights, _, _ = regime_mults(ind, i, cfg)
    votes = row_votes(ind, i, cfg)
    score = score_of(votes, weights)
    bullish = sum(v == 1 for v in votes.values())
    bearish = sum(v == -1 for v in votes.values())
    direction = (1 if score >= entry_threshold and bullish >= MIN_CONFIRMATIONS
                 else -1 if score <= -entry_threshold and bearish >= MIN_CONFIRMATIONS
                 else 0)
    if direction == 0:
        return 0

    quality_ok, _ = quality_filters(ind, i, cfg)
    if not quality_ok:
        return 0

    if mtf_arr is not None and mtf_arr[i] != 0:
        if (direction == 1 and mtf_arr[i] == -1) or (direction == -1 and mtf_arr[i] == 1):
            return 0

    if direction == 1 and pat_bear is not None:
        start = max(3, i - cfg.pattern_lookback + 1)
        if any(bool(pat_bear[j]) for j in range(start, i + 1)):
            return 0
    return direction


def assess_signal(df: pd.DataFrame, sig, cfg: IndicatorSettings | None = None,
                  parent_df: pd.DataFrame | None = None,
                  parent_timeframe: str = "", timeframe: str = "",
                  use_patterns: bool = False, fee_pct: float = 0.1,
                  slippage_pct: float = 0.05) -> dict:
    """Qualify the current signal using conservative historical triple barriers."""
    cfg = cfg or IndicatorSettings()
    horizon = QUALITY_HORIZON_BY_TF.get(timeframe, 3)
    direction = (1 if sig.final in ("BUY", "STRONG BUY") else
                 -1 if sig.final in ("SELL", "STRONG SELL") else 0)
    current_entry = float(sig.close)
    current_tp = float(sig.target_long if direction == 1 else sig.target_short) if direction else None
    current_sl = float(sig.stop_long if direction == 1 else sig.stop_short) if direction else None
    valid_until = None
    if getattr(sig, "timestamp", None) is not None:
        valid_until = sig.timestamp + _timeframe_delta(timeframe) * horizon

    base = {
        "action": "WAIT",
        "grade": "WAIT",
        "direction": "LONG" if direction == 1 else "SHORT" if direction == -1 else "WAIT",
        "status": "TRADE" if direction else "WAIT",
        "probability": None,
        "probability_pct": None,
        "conservative_probability": None,
        "conservative_probability_pct": None,
        "wins": 0,
        "samples": 0,
        "expected_value_r": None,
        "reward_risk": None,
        "valid_for_bars": horizon,
        "valid_until": valid_until,
        "reasons": [],
        "method": "Empirical triple-barrier meta-label",
    }
    if direction == 0:
        base["reasons"] = ["The primary signal is Stable; wait for direction and confirmation."]
        return base

    ind = ta.compute_all(df, cfg)
    mtf_on = _mtf_enabled(timeframe, None)
    mtf_arr = (parent_mtf_array(df, parent_df, parent_timeframe)
               if (mtf_on and parent_timeframe) else None)
    pat_bear = (pattern_flags(df)["pat_bear"].to_numpy() if use_patterns else None)
    warm = warmup_bars(cfg)
    max_start = len(df) - horizon - 1
    wins = 0
    samples = 0

    # Exclude the final horizon bars: their future outcomes are not observable.
    for i in range(warm, max(warm, max_start + 1)):
        historical_direction = _historical_direction(
            df, ind, i, cfg, mtf_arr, pat_bear)
        if historical_direction == 0:
            continue
        atr_value = float(ind["atr"].iloc[i])
        entry = float(df["close"].iloc[i])
        if not np.isfinite(atr_value) or atr_value <= 0 or not np.isfinite(entry):
            continue
        if historical_direction == 1:
            stop = entry - cfg.atr_stop_mult * atr_value
            target = entry + cfg.atr_target_mult * atr_value
        else:
            stop = entry + cfg.atr_stop_mult * atr_value
            target = entry - cfg.atr_target_mult * atr_value
        samples += 1
        wins += _barrier_label(df, i, historical_direction, target, stop, horizon)

    probability = _laplace_probability(wins, samples)
    conservative = _wilson_lower(wins, samples)
    base.update({
        "probability": probability,
        "probability_pct": round(probability * 100.0, 1),
        "conservative_probability": conservative,
        "conservative_probability_pct": round(conservative * 100.0, 1),
        "wins": wins,
        "samples": samples,
    })

    risk = abs(current_entry - current_sl) if current_sl is not None else 0.0
    reward = abs(current_tp - current_entry) if current_tp is not None else 0.0
    rr = reward / risk if risk > 0 else 0.0
    cost_r = ((2.0 * (fee_pct + slippage_pct)) / 100.0 * current_entry / risk
              if risk > 0 else 0.0)
    expected_value = probability * rr - (1.0 - probability) - cost_r
    base["reward_risk"] = round(rr, 2)
    base["expected_value_r"] = round(expected_value, 3)

    reasons: list[str] = []
    if samples < MIN_HISTORICAL_SAMPLES:
        reasons.append(f"Only {samples} historical samples; need at least {MIN_HISTORICAL_SAMPLES}.")
    if probability < MIN_PROBABILITY:
        reasons.append(f"TP-first probability is {probability * 100:.1f}%, below {MIN_PROBABILITY * 100:.0f}%.")
    if conservative < 0.50:
        reasons.append(f"Conservative probability bound is only {conservative * 100:.1f}%.")
    if expected_value <= 0:
        reasons.append(f"Expected value is {expected_value:+.2f}R after estimated costs.")
    if abs(float(sig.score)) < MIN_SCORE:
        reasons.append("Consensus score is not strong enough.")
    if int(sig.agreement) < MIN_AGREEMENT:
        reasons.append(f"Indicator agreement is {sig.agreement}%, below {MIN_AGREEMENT}%.")
    if getattr(sig, "filter_blocked", False):
        reasons.extend(getattr(sig, "filter_reasons", []) or ["A quality filter is active."])
    if getattr(sig, "pattern_vetoed", False):
        reasons.append("A reversal pattern vetoes this setup.")
    if str(getattr(sig, "regime", "")).upper() == "CHOPPY":
        reasons.append("The market regime is choppy.")
    if getattr(sig, "mtf_blocked", False):
        reasons.append("The higher timeframe conflicts with this direction.")

    # Strict trade gate. Grade A requires stronger evidence than the minimum.
    passed = not reasons
    if passed:
        if (probability >= 0.62 and conservative >= 0.50 and expected_value >= 0.15
                and samples >= 40 and abs(float(sig.score)) >= 0.35
                and int(sig.agreement) >= 70):
            grade = "A"
        else:
            grade = "B"
        base["action"] = "TRADE"
        base["grade"] = grade
        base["reasons"] = ["All quality gates passed."]
    else:
        base["grade"] = "WAIT"
        base["reasons"] = reasons
    return base
