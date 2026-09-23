"""
Consensus engine.

Every component casts a vote on the latest CLOSED candle:
    +1  bullish        0  neutral        -1  bearish

Voting components (14 possible):
    12 price/volume indicators  — RSI, MACD, Bollinger, Stochastic, EMA,
                                  Supertrend, ADX, ATR (no vote), OBV, MFI,
                                  CCI, ROC
     2 derivatives sentiment    — funding rate, 24h open-interest change
                                  (live only; dropped when unavailable)

Two filter layers shape the final signal:
    1. REGIME (Choppiness Index) — in a trending market the oscillator
       votes are muted; in a choppy market the trend votes are muted.
    2. MULTI-TIMEFRAME — a directional signal that fights the higher
       timeframe's trend (e.g. a 4h BUY while the daily trend is down)
       is suppressed to NEUTRAL.

Score:
    score = Σ(vote × weight) ÷ Σ(weights of the votes that fired)
    → always in [-1, +1]

    >= +0.50  STRONG BUY        >= +0.25  BUY
    >  -0.25  NEUTRAL           <= -0.25  SELL
    <= -0.50  STRONG SELL

ATR never votes — it only sizes the suggested stop-loss / take-profit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators as ta
from .config import (BUY, MIN_CONFIRMATIONS, MTF_DEFAULT_BY_TF,
                     OSCILLATOR_FAMILY, PARENT_CANDLE_LIMIT,
                     PARENT_EMA_FAST, PARENT_EMA_SLOW, PARENT_TIMEFRAME, SELL,
                     STRONG_BUY, STRONG_SELL, TREND_FAMILY, WEIGHTS,
                     IndicatorSettings, CANDLE_LIMIT)
from .data import TIMEFRAME_SECONDS, drop_unclosed, fetch_derivatives, fetch_ohlcv
from .patterns import PatternState, describe_patterns

# Display order of the 12 core indicators (ATR keeps its slot, weight 0).
INDICATOR_ORDER = ["rsi", "macd", "bb", "stoch", "ema", "supertrend", "adx",
                   "atr", "obv", "mfi", "cci", "roc"]


# ------------------------------------------------------------------- types
@dataclass
class IndicatorVote:
    name: str
    value: str          # human-readable latest value
    vote: int           # +1 / 0 / -1
    weight: float       # weight actually applied (after regime muting)
    reason: str


@dataclass
class Signal:
    symbol: str
    timeframe: str
    timestamp: pd.Timestamp
    close: float
    score: float
    final: str
    bullish: int
    bearish: int
    neutral: int
    agreement: int
    votes: list[IndicatorVote] = field(default_factory=list)
    stop_long: float = 0.0
    target_long: float = 0.0
    stop_short: float = 0.0
    target_short: float = 0.0
    atr: float = 0.0
    atr_pct: float = 0.0
    # filter layers
    regime: str = "—"
    chop: float = float("nan")
    mtf_timeframe: str = ""
    mtf_trend: int = 0        # +1 up / -1 down / 0 unknown
    mtf_blocked: bool = False
    # derivatives
    deriv: dict | None = None
    # candlestick pattern state (entry-quality layer, not a vote)
    patterns: PatternState | None = None


# ------------------------------------------------------------- core votes
def row_votes(ind: pd.DataFrame, i: int, cfg: IndicatorSettings) -> dict[str, int]:
    """
    Votes of the 12 core indicators on candle i.

    Pure function over precomputed indicator columns — used both for the
    live signal (i = last row) and for the backtester (every past row).
    """
    r = ind.iloc[i]
    v: dict[str, int] = {}

    # 1. RSI — mean-reversion
    v["rsi"] = 1 if r["rsi"] < cfg.rsi_os else (-1 if r["rsi"] > cfg.rsi_ob else 0)

    # 2. MACD — trend/momentum direction
    v["macd"] = 1 if r["macd"] > r["macd_signal"] else -1

    # 3. Bollinger %B — price extreme vs volatility bands
    if pd.isna(r["bb_pct_b"]):
        v["bb"] = 0
    elif r["bb_pct_b"] < 0:
        v["bb"] = 1
    elif r["bb_pct_b"] > 1:
        v["bb"] = -1
    else:
        v["bb"] = 0

    # 4. Stochastic — short-term overbought/oversold
    v["stoch"] = 1 if r["stoch_k"] < cfg.stoch_os else (-1 if r["stoch_k"] > cfg.stoch_ob else 0)

    # 5. EMA 9/21 — short-term trend direction
    v["ema"] = 1 if r["ema_fast"] > r["ema_mid"] else -1

    # 6. Supertrend — trailing trend state
    v["supertrend"] = int(r["supertrend_dir"])

    # 7. ADX — only votes when the trend is actually strong
    if pd.isna(r["adx"]) or r["adx"] < cfg.adx_trend_threshold:
        v["adx"] = 0
    else:
        v["adx"] = 1 if r["plus_di"] > r["minus_di"] else -1

    # 8. ATR — volatility gauge, never votes
    v["atr"] = 0

    # 9. OBV — volume flow vs its own 20-EMA
    v["obv"] = 1 if r["obv"] > r["obv_ema"] else -1

    # 10. MFI — volume-weighted overbought/oversold
    v["mfi"] = 1 if r["mfi"] < cfg.mfi_os else (-1 if r["mfi"] > cfg.mfi_ob else 0)

    # 11. CCI — cyclical extreme
    v["cci"] = 1 if r["cci"] < cfg.cci_os else (-1 if r["cci"] > cfg.cci_ob else 0)

    # 12. ROC — momentum with a small deadband to ignore noise
    v["roc"] = 1 if r["roc"] > cfg.roc_deadband else (-1 if r["roc"] < -cfg.roc_deadband else 0)

    return v


def deriv_votes(deriv: dict, cfg: IndicatorSettings) -> dict[str, int]:
    """
    Votes of the derivatives layer (funding + OI) from a live snapshot.

    Only keys whose data is available are returned — a missing feed simply
    does not vote (no dilution of the score) instead of breaking it.
    """
    v: dict[str, int] = {}

    fp = deriv.get("funding_pct")
    if fp is not None:
        if fp >= cfg.funding_hot_pct:
            v["funding"] = -1      # longs paying heavily → crowded long
        elif fp <= -cfg.funding_hot_pct:
            v["funding"] = 1       # shorts paying heavily → crowded short
        else:
            v["funding"] = 0

    oi = deriv.get("oi_chg_24h_pct")
    px = deriv.get("price_chg_24h_pct")
    if oi is not None and px is not None:
        if abs(oi) >= cfg.oi_deadband_pct:
            if oi > 0 and px > 0:
                v["oi"] = 1        # new longs building
            elif oi > 0 and px < 0:
                v["oi"] = -1       # new shorts building
            elif oi < 0 and px < 0:
                v["oi"] = -1       # long unwinding
            else:
                v["oi"] = 0        # short covering — weak, no edge
        else:
            v["oi"] = 0            # below the deadband

    return v


# --------------------------------------------------------------- scoring
def regime_mults(ind: pd.DataFrame, i: int, cfg: IndicatorSettings):
    """
    Regime-aware weights for candle i.

    Returns (weights, chop_value, label):
        TRENDING — oscillators muted × regime_mute
        CHOPPY   — trend followers muted × regime_mute
        MIXED    — no muting
    """
    r = ind.iloc[i]
    chop = r["chop"] if "chop" in r.index else float("nan")
    if pd.isna(chop):
        return dict(WEIGHTS), float("nan"), "—"

    if chop < cfg.chop_trend:
        label = "TRENDING"

        def mult(k: str) -> float:
            return cfg.regime_mute if k in OSCILLATOR_FAMILY else 1.0
    elif chop > cfg.chop_choppy:
        label = "CHOPPY"

        def mult(k: str) -> float:
            return cfg.regime_mute if k in TREND_FAMILY else 1.0
    else:
        label = "MIXED"

        def mult(k: str) -> float:
            return 1.0

    weights = {k: w * mult(k) for k, w in WEIGHTS.items()}
    return weights, float(chop), label


def score_of(votes: dict[str, int], weights: dict[str, float] | None = None) -> float:
    """
    Weighted average of the votes that fired, normalized to [-1, +1].
    The denominator only counts indicators present in `votes`, so a missing
    derivatives feed (or warm-up zeros) never dilutes the score.
    """
    weights = WEIGHTS if weights is None else weights
    active = [k for k in votes if k in weights and weights[k] > 0]
    total_w = sum(weights[k] for k in active)
    if total_w == 0:
        return 0.0
    return sum(weights[k] * votes[k] for k in active) / total_w


# ------------------------------------------------------------- MTF logic
def parent_trend(df_parent: pd.DataFrame | None) -> int:
    """Trend of the higher timeframe: +1 up / -1 down / 0 unknown."""
    if df_parent is None or len(df_parent) < PARENT_EMA_SLOW + 5:
        return 0
    c = df_parent["close"]
    fast = ta.ema(c, PARENT_EMA_FAST).iloc[-1]
    slow = ta.ema(c, PARENT_EMA_SLOW).iloc[-1]
    return 1 if fast > slow else -1


def parent_mtf_array(df: pd.DataFrame, df_parent: pd.DataFrame | None,
                     parent_tf: str) -> np.ndarray:
    """
    For every candle of `df`, the higher-timeframe trend at that moment.

    Only PARENT CANDLES THAT WERE ALREADY CLOSED at the signal candle's open
    time count — this keeps the backtest free of look-ahead bias.
    """
    arr = np.zeros(len(df), dtype=int)
    if df_parent is None:
        return arr
    tf_s = TIMEFRAME_SECONDS.get(parent_tf)
    if tf_s is None or len(df_parent) < PARENT_EMA_SLOW + 5:
        return arr

    c = df_parent["close"]
    fast = ta.ema(c, PARENT_EMA_FAST)
    slow = ta.ema(c, PARENT_EMA_SLOW)
    trend = pd.Series(np.where(fast > slow, 1, -1), index=df_parent.index)
    trend = trend.where(slow.notna(), 0).to_numpy()

    parent_close_ts = (df_parent.index + pd.Timedelta(seconds=tf_s)).asi8 // 1_000_000
    sig_ts = df.index.asi8 // 1_000_000
    j = np.searchsorted(parent_close_ts, sig_ts, side="right") - 1
    valid = j >= 0
    arr[valid] = trend[j[valid]]
    return arr


# ------------------------------------------------------------- formatting
def _num(x, nd: int = 2) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:,.{nd}f}"


def _deriv_reasons(deriv: dict, cfg: IndicatorSettings):
    fp = deriv.get("funding_pct")
    if fp is not None:
        if fp >= cfg.funding_hot_pct:
            f_reason = f"Crowded longs (funding ≥ +{cfg.funding_hot_pct:.2f}%) — contrarian short"
        elif fp <= -cfg.funding_hot_pct:
            f_reason = f"Crowded shorts (funding ≤ −{cfg.funding_hot_pct:.2f}%) — contrarian long"
        else:
            f_reason = "Funding normal — no positioning edge"
    else:
        f_reason = "Funding unavailable"

    oi = deriv.get("oi_chg_24h_pct")
    px = deriv.get("price_chg_24h_pct")
    if oi is None:
        o_reason = "OI history unavailable on this source"
    elif abs(oi) < cfg.oi_deadband_pct:
        o_reason = "OI change below deadband"
    elif oi > 0 and px > 0:
        o_reason = "Price↑ OI↑ — new longs building"
    elif oi > 0 and px < 0:
        o_reason = "Price↓ OI↑ — new shorts building"
    elif oi < 0 and px < 0:
        o_reason = "Price↓ OI↓ — long unwinding"
    else:
        o_reason = "Price↑ OI↓ — short covering (weak)"
    return f_reason, o_reason


# ------------------------------------------------------------- evaluate
def evaluate(df: pd.DataFrame, ind: pd.DataFrame, cfg: IndicatorSettings | None = None,
             symbol: str = "", timeframe: str = "", deriv: dict | None = None,
             mtf_trend: int = 0, mtf_timeframe: str = "",
             patterns: PatternState | None = None) -> Signal:
    """
    Build the full signal for the last candle:
    12 indicator votes + (optional) 2 derivatives votes,
    regime-aware weights, consensus score, MTF suppression, ATR levels.
    """
    cfg = cfg or IndicatorSettings()
    i = len(ind) - 1
    r = ind.iloc[i]
    weights, chop, regime = regime_mults(ind, i, cfg)

    raw = row_votes(ind, i, cfg)
    if deriv:
        raw.update(deriv_votes(deriv, cfg))

    close = float(df["close"].iloc[i])
    atr_v = float(r["atr"])
    atr_pct = float(r["atr_pct"])
    golden = r["ema_trend_fast"] > r["ema_trend_slow"]

    votes: list[IndicatorVote] = []

    def add(key: str, label: str, value: str, reason: str) -> None:
        votes.append(IndicatorVote(name=label, value=value, vote=raw[key],
                                   weight=weights.get(key, 0.0), reason=reason))

    add("rsi", "RSI (14)", _num(r["rsi"], 1),
        "Oversold (< 30)" if raw["rsi"] == 1 else
        "Overbought (> 70)" if raw["rsi"] == -1 else "Neutral zone")

    add("macd", "MACD (12,26,9)",
        f"MACD {_num(r['macd'], 1)} · sig {_num(r['macd_signal'], 1)}",
        "MACD above signal line" if raw["macd"] == 1 else "MACD below signal line")

    add("bb", "Bollinger (20, 2)", f"%B {_num(r['bb_pct_b'])}",
        "Below lower band → reversion up" if raw["bb"] == 1 else
        "Above upper band → reversion down" if raw["bb"] == -1 else "Inside the bands")

    add("stoch", "Stochastic (14,3)",
        f"K {_num(r['stoch_k'], 1)} · D {_num(r['stoch_d'], 1)}",
        "Oversold (< 20)" if raw["stoch"] == 1 else
        "Overbought (> 80)" if raw["stoch"] == -1 else "Middle zone")

    ema_reason = ("EMA9 above EMA21 — short-term trend up" if raw["ema"] == 1
                  else "EMA9 below EMA21 — short-term trend down")
    ema_reason += " · golden cross active (50>200)" if golden else " · death cross active (50<200)"
    add("ema", "EMA (9/21 · 50/200)",
        f"EMA9 {_num(r['ema_fast'], 1)} · EMA21 {_num(r['ema_mid'], 1)}", ema_reason)

    add("supertrend", "Supertrend (10, 3)",
        f"line {_num(r['supertrend'], 1)}",
        "Price above Supertrend — trend long" if raw["supertrend"] == 1 else
        "Price below Supertrend — trend short" if raw["supertrend"] == -1
        else "Warm-up / no state")

    add("adx", "ADX (14)",
        f"ADX {_num(r['adx'], 1)} · +DI {_num(r['plus_di'], 1)} · −DI {_num(r['minus_di'], 1)}",
        "Strong uptrend (ADX ≥ 25, +DI leads)" if raw["adx"] == 1 else
        "Strong downtrend (ADX ≥ 25, −DI leads)" if raw["adx"] == -1
        else "No strong trend (ADX < 25)")

    add("atr", "ATR (14)", f"{_num(atr_v)} ({atr_pct:.2f}% of price)",
        "Volatility gauge — sizes the stops, never votes")

    add("obv", "OBV (vs 20-EMA)", "above trend" if raw["obv"] == 1 else "below trend",
        "Volume flow confirms the uptrend" if raw["obv"] == 1
        else "Volume flow confirms the downtrend")

    add("mfi", "MFI (14)", _num(r["mfi"], 1),
        "Money flow oversold (< 20)" if raw["mfi"] == 1 else
        "Money flow overbought (> 80)" if raw["mfi"] == -1 else "Money flow neutral")

    add("cci", "CCI (20)", _num(r["cci"], 1),
        "Oversold (< −100)" if raw["cci"] == 1 else
        "Overbought (> +100)" if raw["cci"] == -1 else "Normal range")

    add("roc", "ROC (10)", f"{r['roc']:+.2f}%",
        "Positive momentum" if raw["roc"] == 1 else
        "Negative momentum" if raw["roc"] == -1 else "Momentum flat (deadband)")

    if deriv:
        f_reason, o_reason = _deriv_reasons(deriv, cfg)
        fp = deriv.get("funding_pct")
        votes.append(IndicatorVote(
            name="Funding (derivatives)",
            value=f"{fp:+.4f}% via {deriv.get('source', '?')}" if fp is not None else "unavailable",
            vote=raw.get("funding", 0),
            weight=weights.get("funding", 0.0) if "funding" in raw else 0.0,
            reason=f_reason))
        oi = deriv.get("oi_chg_24h_pct")
        px = deriv.get("price_chg_24h_pct")
        oi_str = (f"OI {oi:+.1f}% · px {px:+.1f}% (24h)"
                  if oi is not None and px is not None else "unavailable")
        votes.append(IndicatorVote(
            name="Open Interest Δ (24h)",
            value=oi_str,
            vote=raw.get("oi", 0),
            weight=weights.get("oi", 0.0) if "oi" in raw else 0.0,
            reason=o_reason))

    # ------------------------------------------------------------ consensus
    bullish = sum(1 for x in votes if x.vote == 1)
    bearish = sum(1 for x in votes if x.vote == -1)
    neutral = len(votes) - bullish - bearish
    score = score_of(raw, weights)

    if score >= STRONG_BUY and bullish >= MIN_CONFIRMATIONS:
        final = "STRONG BUY"
    elif score >= BUY and bullish >= MIN_CONFIRMATIONS:
        final = "BUY"
    elif score <= STRONG_SELL and bearish >= MIN_CONFIRMATIONS:
        final = "STRONG SELL"
    elif score <= SELL and bearish >= MIN_CONFIRMATIONS:
        final = "SELL"
    else:
        final = "NEUTRAL"

    # ----------------------------------------------- MTF confluence layer
    mtf_blocked = False
    if mtf_trend == -1 and final in ("BUY", "STRONG BUY"):
        final, mtf_blocked = "NEUTRAL", True
    elif mtf_trend == 1 and final in ("SELL", "STRONG SELL"):
        final, mtf_blocked = "NEUTRAL", True

    voting = bullish + bearish
    agreement = round(100.0 * max(bullish, bearish) / voting) if voting else 0

    return Signal(
        symbol=symbol, timeframe=timeframe, timestamp=ind.index[i],
        close=close, score=round(score, 4), final=final,
        bullish=bullish, bearish=bearish, neutral=neutral, agreement=agreement,
        votes=votes,
        stop_long=close - cfg.atr_stop_mult * atr_v,
        target_long=close + cfg.atr_target_mult * atr_v,
        stop_short=close + cfg.atr_stop_mult * atr_v,
        target_short=close - cfg.atr_target_mult * atr_v,
        atr=atr_v, atr_pct=atr_pct,
        regime=regime, chop=chop,
        mtf_timeframe=mtf_timeframe, mtf_trend=mtf_trend, mtf_blocked=mtf_blocked,
        deriv=deriv, patterns=patterns,
    )


# --------------------------------------------------------- full pipeline
def full_signal(symbol: str, timeframe: str, limit: int = CANDLE_LIMIT,
                use_mtf: bool | None = None, use_deriv: bool = True,
                cfg: IndicatorSettings | None = None) -> Signal:
    """
    Live signal pipeline:
      fetch candles (drop the still-forming one) → compute indicators →
      fetch parent-timeframe trend (MTF) → fetch derivatives sentiment →
      evaluate with all filter layers.

    use_mtf=None → per-timeframe default from MTF_DEFAULT_BY_TF
    (walk-forward evidence: on for 15m/1h/4h, off for 1d).
    """
    cfg = cfg or IndicatorSettings()
    df = drop_unclosed(fetch_ohlcv(symbol, timeframe, limit), timeframe)
    ind = ta.compute_all(df, cfg)

    mtf_timeframe = PARENT_TIMEFRAME.get(timeframe, "")
    if use_mtf is None:
        use_mtf = MTF_DEFAULT_BY_TF.get(timeframe, True)
    mtf_trend = 0
    if use_mtf and mtf_timeframe:
        p = drop_unclosed(fetch_ohlcv(symbol, mtf_timeframe, PARENT_CANDLE_LIMIT),
                          mtf_timeframe)
        mtf_trend = parent_trend(p)

    deriv = fetch_derivatives(symbol, df) if use_deriv else None
    patterns = describe_patterns(df, i=None, lookback=cfg.pattern_lookback)

    return evaluate(df, ind, cfg, symbol, timeframe, deriv=deriv,
                    mtf_trend=mtf_trend, mtf_timeframe=mtf_timeframe,
                    patterns=patterns)
