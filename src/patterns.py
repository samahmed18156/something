"""
Candlestick pattern detection — strict, deterministic rules.

Patterns are NOT a 15th vote: they are an entry QUALITY layer. The only
action they take in the engine is a VETO — a fresh bearish pattern on the
last closed candle(s) blocks a new long entry. Bullish patterns tag the
signal as "confirmed" (display only). Whether that veto earns its keep is
decided by the A/B walk-forward comparison, not by belief.

Implemented (classic definitions, no gaps required):
    bullish / bearish engulfing
    hammer · shooting star (with trend context from the prior 3 closes)
    doji (indecision flag)
    three white soldiers · three black crows
    morning star · evening star
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PATTERN_COLUMNS = [
    "bull_engulf", "bear_engulf", "hammer", "shooting_star", "doji",
    "three_soldiers", "three_crows", "morning_star", "evening_star",
]

NAMES = {
    "bull_engulf": "Bullish engulfing",
    "bear_engulf": "Bearish engulfing",
    "hammer": "Hammer",
    "shooting_star": "Shooting star",
    "doji": "Doji",
    "three_soldiers": "Three white soldiers",
    "three_crows": "Three black crows",
    "morning_star": "Morning star",
    "evening_star": "Evening star",
}

BULLISH_PATTERNS = ["bull_engulf", "hammer", "three_soldiers", "morning_star"]
BEARISH_PATTERNS = ["bear_engulf", "shooting_star", "three_crows", "evening_star"]


@dataclass
class PatternState:
    bullish: bool = False
    bearish: bool = False
    doji: bool = False
    names: list[str] = field(default_factory=list)


def pattern_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-bar pattern booleans for the whole OHLCV frame (vectorized where
    possible). The first three rows are warm-up and always False.
    """
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)

    body = np.abs(c - o)
    rng = h - l
    rng_s = np.where(rng == 0, np.nan, rng)
    upper = h - np.maximum(o, c)        # upper wick, ≥ 0
    lower = np.minimum(o, c) - l        # lower wick, ≥ 0
    body_pct = body / rng_s

    bull = c > o
    bear = c < o

    r1 = lambda s: np.roll(s, 1)   # noqa: E731
    r2 = lambda s: np.roll(s, 2)   # noqa: E731
    r3 = lambda s: np.roll(s, 3)   # noqa: E731

    flags = pd.DataFrame(index=df.index, dtype=bool)

    # --- engulfing: today's body engulfs yesterday's opposite body ---
    flags["bull_engulf"] = (
        bull & r1(bear)
        & (o <= r1(c)) & (c >= r1(o))
        & (body > r1(body))
    )
    flags["bear_engulf"] = (
        bear & r1(bull)
        & (o >= r1(c)) & (c <= r1(o))
        & (body > r1(body))
    )

    # --- doji: body ≤ 10% of the range ---
    flags["doji"] = (body_pct <= 0.10)

    # --- hammer: long lower shadow after a decline (prior 3 closes down) ---
    flags["hammer"] = (
        (lower >= 2.0 * body) & (upper <= body) & (body > 0)
        & (c < r3(c))
    )

    # --- shooting star: long upper shadow after a rise ---
    flags["shooting_star"] = (
        (upper >= 2.0 * body) & (lower <= body) & (body > 0)
        & (c > r3(c))
    )

    # --- three soldiers / three crows: 3 consecutive big, marching bodies ---
    big = body_pct >= 0.50
    flags["three_soldiers"] = (
        bull & r1(bull) & r2(bull)
        & (c > r1(c)) & (r1(c) > r2(c))
        & big & r1(big) & r2(big)
    )
    flags["three_crows"] = (
        bear & r1(bear) & r2(bear)
        & (c < r1(c)) & (r1(c) < r2(c))
        & big & r1(big) & r2(big)
    )

    # --- morning / evening star: big body, small star, decisive 3rd ---
    third_body = r2(body)
    mid_of_first = (r2(o) + r2(c)) / 2.0
    star_small = r1(body) <= 0.30 * third_body
    flags["morning_star"] = (
        r2(bear) & star_small & bull
        & (c > mid_of_first) & (third_body > 0)
    )
    flags["evening_star"] = (
        r2(bull) & star_small & bear
        & (c < mid_of_first) & (third_body > 0)
    )

    flags.iloc[:3] = False  # warm-up (np.roll wrap-around)

    flags["pat_bull"] = flags[BULLISH_PATTERNS].any(axis=1)
    flags["pat_bear"] = flags[BEARISH_PATTERNS].any(axis=1)
    return flags


def describe_patterns(df: pd.DataFrame, i: int | None = None,
                      lookback: int = 1) -> PatternState:
    """
    Human-readable pattern state for candle i (default: last) scanning the
    previous `lookback` closed candles.
    """
    flags = pattern_flags(df)
    i = len(df) - 1 if i is None else i
    start = max(3, i - lookback + 1)

    state = PatternState()
    for j in range(start, i + 1):
        row = flags.iloc[j]
        for col, label in NAMES.items():
            if row[col] and label not in state.names:
                state.names.append(label)
                if col == "doji":
                    state.doji = True
                elif col in BULLISH_PATTERNS:
                    state.bullish = True
                elif col in BEARISH_PATTERNS:
                    state.bearish = True
    return state
