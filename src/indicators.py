"""
The 10 core technical indicators — pure pandas/numpy implementations
(no TA-Lib build step required).

Every function returns a Series/DataFrame aligned to the input candle
index, so all columns line up 1:1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import IndicatorSettings


# ---------------------------------------------------------------- helpers
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


# ------------------------------------------------------------------- 1 RSI
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)   # all-up candles
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)   # flat candles
    return out


# ------------------------------------------------------------------ 2 MACD
def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": macd_line - signal_line,
    })


# --------------------------------------------------------- 3 Bollinger BBs
def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, period)
    std = close.rolling(period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame({
        "bb_mid": mid,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_pct_b": (close - lower) / width,
        "bb_bandwidth": width / mid.replace(0.0, np.nan),
    })


# ---------------------------------------------------------- 4 Stochastic
def stochastic(df: pd.DataFrame, period: int = 14, smooth_k: int = 3,
               smooth_d: int = 3) -> pd.DataFrame:
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    rng = (high_max - low_min).replace(0.0, np.nan)
    raw_k = 100.0 * (df["close"] - low_min) / rng
    k = sma(raw_k, smooth_k)
    d = sma(k, smooth_d)
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


# ----------------------------------------------------------------- 6 ADX
def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder's ADX with +DI / −DI."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_ = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False,
                                  min_periods=period).mean() / atr_
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False,
                                    min_periods=period).mean() / atr_
    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_ = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_})


# ------------------------------------------------------------------- 7 ATR
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


# ------------------------------------------------------------------- 8 OBV
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


# ------------------------------------------------------------------- 9 CCI
def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = sma(tp, period)
    mean_dev = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - ma) / (0.015 * mean_dev.replace(0.0, np.nan))


# ------------------------------------------------------------------ 10 ROC
def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change, in percent."""
    return close.pct_change(period) * 100.0


# ------------------------------------------------------- 11 Supertrend -----
def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.DataFrame:
    """
    Classic Supertrend (ATR trailing channel).

    Returns:
        supertrend      — the support/resistance line (float)
        supertrend_dir  — +1 while price rides above (bullish),
                          -1 below (bearish), 0 during warm-up
    """
    atr_ = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    ub = (hl2 + mult * atr_).to_numpy()
    lb = (hl2 - mult * atr_).to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    fub = np.full(n, np.nan)
    flb = np.full(n, np.nan)
    trend = np.zeros(n, dtype=int)

    # Start at the first bar where the ATR (hence the bands) is valid —
    # carrying NaN forward from bar 0 would poison the whole series.
    valid = ~np.isnan(ub)
    first = int(np.argmax(valid)) if valid.any() else n

    if first < n:
        fub[first] = ub[first]
        flb[first] = lb[first]
        trend[first] = 1 if close[first] > flb[first] else -1
        for i in range(first + 1, n):
            fub[i] = ub[i] if (ub[i] < fub[i - 1] or close[i - 1] > fub[i - 1]) else fub[i - 1]
            flb[i] = lb[i] if (lb[i] > flb[i - 1] or close[i - 1] < flb[i - 1]) else flb[i - 1]
            if trend[i - 1] == 1:
                trend[i] = -1 if close[i] < flb[i] else 1
            else:
                trend[i] = 1 if close[i] > fub[i] else -1

    line = np.where(trend == -1, fub, flb)
    line = np.where(trend == 0, np.nan, line)
    return pd.DataFrame({
        "supertrend": line,
        "supertrend_dir": trend,
    }, index=df.index)


# ------------------------------------------------------ 12 MFI ------------
def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index — volume-weighted RSI."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mf = tp * df["volume"]
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(period).sum()
    neg_sum = neg.rolling(period).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + ratio)
    out = out.mask((neg_sum == 0) & (pos_sum > 0), 100.0)
    return out


# -------------------------------------------------- regime: Choppiness -----
def choppiness(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Choppiness Index (CHOP): 100 = fully random/choppy, 10 = perfect trend.

    < 38.2 trending · > 61.8 choppy (common trading cutoffs).
    """
    atr_sum = atr(df, period).rolling(period).sum()
    hh = df["high"].rolling(period).max()
    ll = df["low"].rolling(period).min()
    rng = (hh - ll).replace(0.0, np.nan)
    return 100.0 * np.log(atr_sum / rng) / np.log(period)


# ------------------------------------------------------------- compute all
def compute_all(df: pd.DataFrame, cfg: IndicatorSettings | None = None) -> pd.DataFrame:
    """Compute every indicator for an OHLCV DataFrame in one pass."""
    cfg = cfg or IndicatorSettings()
    ind = pd.DataFrame(index=df.index)
    ind["close"] = df["close"]
    ind["volume"] = df["volume"]

    ind["rsi"] = rsi(df["close"], cfg.rsi_period)
    ind = ind.join(macd(df["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal))
    ind = ind.join(bollinger(df["close"], cfg.bb_period, cfg.bb_std))
    ind = ind.join(stochastic(df, cfg.stoch_period, cfg.stoch_smooth, cfg.stoch_smooth))
    ind["ema_fast"] = ema(df["close"], cfg.ema_fast)
    ind["ema_mid"] = ema(df["close"], cfg.ema_mid)
    ind["ema_trend_fast"] = ema(df["close"], cfg.ema_trend_fast)
    ind["ema_trend_slow"] = ema(df["close"], cfg.ema_trend_slow)
    ind = ind.join(adx(df, cfg.adx_period))
    ind["atr"] = atr(df, cfg.atr_period)
    ind["atr_pct"] = 100.0 * ind["atr"] / df["close"].replace(0.0, np.nan)
    ind["obv"] = obv(df["close"], df["volume"])
    ind["obv_ema"] = ema(ind["obv"], cfg.obv_ema)
    ind["cci"] = cci(df, cfg.cci_period)
    ind["roc"] = roc(df["close"], cfg.roc_period)
    ind = ind.join(supertrend(df, cfg.supertrend_period, cfg.supertrend_mult))
    ind["mfi"] = mfi(df, cfg.mfi_period)
    ind["chop"] = choppiness(df, cfg.chop_period)
    return ind
