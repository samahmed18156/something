"""
Market data — Binance public REST API (no API keys required).

Binance blocks some regions/IP ranges (HTTP 451), so we walk a fallback
chain of Binance public endpoints first and then fall back to OKX via ccxt.

Endpoints used (public, keyless):
    https://<host>/api/v3/klines?symbol=BTCUSDT&interval=4h&limit=500
"""
from __future__ import annotations

import json
import urllib.request

import ccxt
import pandas as pd

BINANCE_HOSTS = [
    "api.binance.com",
    "data-api.binance.vision",   # official public market-data mirror
    "api1.binance.com",
    "api2.binance.com",
    "api3.binance.com",
]

USER_AGENT = "crypto-signal-system/1.0"
_okx: ccxt.okx | None = None

TIMEFRAME_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "8h": 28800,
    "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800,
}


def _binance_klines(pair: str, interval: str, limit: int,
                    since_ms: int | None = None) -> list:
    """Raw klines from the first reachable Binance host (raises if none)."""
    errors: list[str] = []
    for host in BINANCE_HOSTS:
        url = (f"https://{host}/api/v3/klines"
               f"?symbol={pair}&interval={interval}&limit={limit}")
        if since_ms is not None:
            url += f"&startTime={since_ms}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            errors.append(f"{host}: {type(e).__name__}")
    raise RuntimeError("Binance klines failed: " + " | ".join(errors))


def fetch_ohlcv(symbol: str, timeframe: str = "4h", limit: int = 500) -> pd.DataFrame:
    """
    Fetch OHLCV candles for a ccxt-format symbol (e.g. "BTC/USDT").

    ``timeframe`` uses Binance interval strings: 15m, 1h, 4h, 1d (…).

    Returns a DataFrame indexed by candle open time (UTC) with columns:
    open, high, low, close, volume
    """
    global _okx
    limit = int(min(limit, 1000))
    pair = symbol.replace("/", "")          # BTC/USDT -> BTCUSDT
    errors: list[str] = []

    try:
        raw = _binance_klines(pair, timeframe, limit)
        if raw:
            return _to_df(raw)
    except Exception as e:
        errors.append(str(e))

    # Last-resort fallback: OKX spot (same USDT pairs, 300-candle cap).
    try:
        if _okx is None:
            _okx = ccxt.okx({"enableRateLimit": True})
        raw = _okx.fetch_ohlcv(symbol, timeframe=timeframe, limit=min(limit, 300))
        if raw:
            return _to_df(raw)
    except Exception as e:
        errors.append(f"okx: {type(e).__name__}")

    raise RuntimeError(
        "Could not fetch market data. Tried: " + " | ".join(errors)
        + " — check your internet connection."
    )


def fetch_ohlcv_paged(symbol: str, timeframe: str = "4h",
                      total: int = 3000) -> pd.DataFrame:
    """
    Page back beyond Binance's 1000-candle cap using ``startTime``.

    Use this for long backtests / walk-forward validation
    (e.g. 4000×1h ≈ 6.6 months of history). Binance-only — the OKX
    fallback has no simple paging endpoint.
    """
    pair = symbol.replace("/", "")
    tf_ms = TIMEFRAME_SECONDS[timeframe] * 1000
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    since = now_ms - (int(total) + 10) * tf_ms

    rows: list = []
    while True:
        batch = _binance_klines(pair, timeframe, 1000, since)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        since = batch[-1][0] + tf_ms
        if since >= now_ms or len(rows) >= int(total) + 10:
            break

    df = _to_df(rows).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.tail(int(total))


def _to_df(raw: list) -> pd.DataFrame:
    # Binance klines rows carry 12 fields (quote volume, trade counts, …) —
    # keep the first 6: open time, open, high, low, close, volume.
    if raw and len(raw[0]) > 6:
        raw = [row[:6] for row in raw]
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("datetime")[["open", "high", "low", "close", "volume"]]


def drop_unclosed(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Drop the last candle if it is still forming (its close time is in the
    future). Signals and backtests must only act on CLOSED candles, otherwise
    "repainting" makes live results look better than history.
    """
    tf_s = TIMEFRAME_SECONDS.get(timeframe)
    if tf_s is None or df.empty:
        return df
    last_close = int(df.index[-1].timestamp()) + tf_s
    if last_close > int(pd.Timestamp.now(tz="UTC").timestamp()):
        return df.iloc[:-1]
    return df


def fetch_derivatives(symbol: str, df: pd.DataFrame | None = None) -> dict | None:
    """
    Live derivatives sentiment — public endpoints, no API keys:
        funding rate        (Binance futures, OKX fallback)
        24h open-interest   (Binance futures only; OKX has no OI history)
        24h price change    (from the spot candles we already have)

    Returns a dict or None when no derivatives source is reachable.
    """
    pair = symbol.replace("/", "")
    swap = symbol.replace("/", "-") + "-SWAP"
    funding = None
    oi_chg = None
    source = None

    try:
        req = urllib.request.Request(
            f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={pair}",
            headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            funding = float(json.loads(resp.read().decode())["lastFundingRate"])
        try:
            req = urllib.request.Request(
                f"https://fapi.binance.com/fapi/v1/openInterestHist"
                f"?symbol={pair}&period=1h&limit=25",
                headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=10) as resp:
                rows = json.loads(resp.read().decode())
            if len(rows) >= 2:
                oi0 = float(rows[0]["sumOpenInterest"])
                oi1 = float(rows[-1]["sumOpenInterest"])
                if oi0 > 0:
                    oi_chg = (oi1 / oi0 - 1.0) * 100.0
        except Exception:
            pass  # OI history unavailable — funding still usable
        source = "binance"
    except Exception:
        # OKX fallback: funding only
        try:
            req = urllib.request.Request(
                f"https://www.okx.com/api/v5/public/funding-rate?instId={swap}",
                headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=10) as resp:
                d = json.loads(resp.read().decode())
            funding = float(d["data"][0]["fundingRate"])
            source = "okx"
        except Exception:
            return None

    if funding is None:
        return None

    price_24h = None
    if df is not None and len(df) > 2:
        now = pd.Timestamp.now(tz="UTC")
        target = now - pd.Timedelta(hours=24)
        i = min(df.index.searchsorted(target), len(df) - 1)
        if i >= 0 and df.index[i] <= now:
            price_24h = (float(df["close"].iloc[-1]) / float(df["close"].iloc[i]) - 1.0) * 100.0

    return {
        "funding_pct": funding * 100.0,
        "oi_chg_24h_pct": oi_chg,
        "price_chg_24h_pct": price_24h,
        "source": source,
    }
