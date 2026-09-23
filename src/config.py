"""
Central configuration for the Crypto Signal System.

Tweak anything here — coins, timeframes, indicator periods, consensus
weights, regime/MTF/derivatives settings and signal thresholds — without
touching the engine code.
"""
from dataclasses import dataclass

# ---------------------------------------------------------------- coins ---
# Grouped so the dashboard can offer quick-selects. Any Binance spot pair
# works — just add "COIN/USDT" to the right list.
MAJORS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
]
ALT_MAJORS = [
    "ADA/USDT",
    "DOGE/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "DOT/USDT",
    "LTC/USDT",
    "TRX/USDT",
    "TON/USDT",
]
TRENDING = [
    "SUI/USDT",
    "APT/USDT",
    "NEAR/USDT",
    "AAVE/USDT",
    "PEPE/USDT",
]

ALL_COINS = MAJORS + ALT_MAJORS + TRENDING
DEFAULT_COINS = MAJORS + ALT_MAJORS   # trenders available via the sidebar

DEFAULT_TIMEFRAME = "4h"
TIMEFRAMES = ["15m", "1h", "4h", "1d"]

CANDLE_LIMIT = 500          # candles fetched per coin (max 1000 on Binance)
PARENT_CANDLE_LIMIT = 500   # candles for the higher timeframe (MTF filter)

# ------------------------------------------------------ indicator params ---
@dataclass
class IndicatorSettings:
    # 1. RSI
    rsi_period: int = 14
    rsi_ob: float = 70.0    # overbought
    rsi_os: float = 30.0    # oversold
    # 2. MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # 3. Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    # 4. Stochastic
    stoch_period: int = 14
    stoch_smooth: int = 3
    stoch_ob: float = 80.0
    stoch_os: float = 20.0
    # 5. EMA crossovers
    ema_fast: int = 9
    ema_mid: int = 21
    ema_trend_fast: int = 50
    ema_trend_slow: int = 200
    # 6. ADX
    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    # 7. ATR (volatility → stop/target sizing, never votes)
    atr_period: int = 14
    atr_stop_mult: float = 1.5
    atr_target_mult: float = 2.0
    # 8. OBV
    obv_ema: int = 20
    # 9. CCI
    cci_period: int = 20
    cci_ob: float = 100.0
    cci_os: float = -100.0
    # 10. ROC
    roc_period: int = 10
    roc_deadband: float = 0.5   # percent — ignore moves smaller than this
    # 11. Supertrend
    supertrend_period: int = 10
    supertrend_mult: float = 3.0
    # 12. MFI (Money Flow Index)
    mfi_period: int = 14
    mfi_ob: float = 80.0
    mfi_os: float = 20.0

    # --- regime filter (Choppiness Index) ---
    chop_period: int = 14
    chop_trend: float = 38.2     # CHOP below → trending market
    chop_choppy: float = 61.8    # CHOP above → choppy/ranging market
    regime_mute: float = 0.4     # weight multiplier for the muted family

    # --- derivatives sentiment (live only) ---
    funding_hot_pct: float = 0.10   # |funding| ≥ 0.10% → crowded positioning
    oi_deadband_pct: float = 3.0    # ignore OI changes smaller than 3% / 24h

    # --- candlestick pattern entry filter ---
    pattern_lookback: int = 1       # closed candles scanned for a fresh veto
                                    # pattern (1 = last closed candle only)


# Weights per indicator family for the regime filter.
TREND_FAMILY = {"macd", "ema", "supertrend", "adx", "roc"}
OSCILLATOR_FAMILY = {"rsi", "stoch", "cci", "bb", "mfi"}

# ------------------------------------------------------------- consensus ---
# How much each indicator counts. ATR has weight 0 on purpose: it is a
# volatility gauge, not a directional indicator. Funding/OI are live-only
# derivatives votes (weight 0 contribution when data is unavailable).
WEIGHTS = {
    "macd": 1.5,
    "ema": 1.5,
    "supertrend": 1.25,
    "adx": 1.25,
    "obv": 1.25,
    "roc": 1.0,
    "rsi": 1.0,
    "stoch": 1.0,
    "cci": 1.0,
    "bb": 1.0,
    "mfi": 1.0,
    "funding": 1.0,
    "oi": 1.0,
    "atr": 0.0,
}

# Signal thresholds on the normalized score in [-1, +1].
STRONG_BUY = 0.50
BUY = 0.25
SELL = -0.25
STRONG_SELL = -0.50

# A directional signal also needs at least this many agreeing votes,
# otherwise it is downgraded to NEUTRAL (noise filter).
MIN_CONFIRMATIONS = 3

# ------------------------------------------------- multi-timeframe filter ---
# Higher timeframe whose trend must not be fought by the signal.
PARENT_TIMEFRAME = {"15m": "1h", "1h": "4h", "4h": "1d", "1d": "1w"}
PARENT_EMA_FAST = 21   # parent-trend EMAs (faster / slower)
PARENT_EMA_SLOW = 50

# Default MTF confluence per timeframe — set by walk-forward evidence:
# 4h benefits strongly, 1h marginally, 1d is hurt (the weekly parent trend
# is too slow; the 1d edge comes from buying dips against it).
MTF_DEFAULT_BY_TF = {"15m": True, "1h": True, "4h": True, "1d": False}
