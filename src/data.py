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

# Symbols that are not useful as directional altcoin choices in the Spot
# signal board: stablecoins, fiat-like tokens, and leveraged-token suffixes.
_STABLE_BASES = {
    "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USDP", "USD1",
    "USDE", "USDD", "PYUSD", "FRAX", "LUSD", "RLUSD", "UST", "EUR",
    "TRY", "BRL", "GBP", "UAH", "RUB", "BIDR", "NGN", "ARS", "ZAR",
}
_LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


def fetch_top_usdt_spot_symbols(limit: int = 100) -> list[str]:
    """Return the most liquid public Binance USDT altcoin spot pairs.

    Ranking is based on current 24-hour quote volume, not a permanent list.
    This keeps the menu relevant while excluding stablecoin bases, leveraged
    tokens, and BTC itself. It intentionally does not change the strategy or
    enable any automated execution.
    """
    errors: list[str] = []
    for host in BINANCE_HOSTS:
        try:
            url = f"https://{host}/api/v3/ticker/24hr"
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as response:
                rows = json.loads(response.read().decode())
            candidates = []
            for row in rows:
                symbol = str(row.get("symbol", ""))
                if not symbol.endswith("USDT"):
                    continue
                base = symbol[:-4]
                if (base in _STABLE_BASES or base == "BTC" or base.startswith("1000")
                        or base.endswith(_LEVERAGED_SUFFIXES)):
                    continue
                try:
                    quote_volume = float(row.get("quoteVolume", 0.0))
                except (TypeError, ValueError):
                    quote_volume = 0.0
                if quote_volume <= 0:
                    continue
                candidates.append((quote_volume, f"{base}/USDT"))
            candidates.sort(reverse=True)
            output = []
            seen = set()
            for _, pair in candidates:
                if pair not in seen:
                    output.append(pair)
                    seen.add(pair)
                if len(output) >= int(limit):
                    break
            if output:
                return output
            raise RuntimeError("no eligible USDT spot pairs returned")
        except Exception as exc:
            errors.append(f"{host}: {type(exc).__name__}")
    raise RuntimeError("Top liquid altcoin list unavailable: " + " | ".join(errors))


def normalize_utc_index(index: pd.Index) -> pd.DatetimeIndex:
    """Normalize exchange timestamps for pandas 2.x/3.x compatibility.

    Newer pandas versions may preserve a seconds/milliseconds datetime unit
    from API data. Some arithmetic and integer conversions then attempt a
    lossless unit cast and raise ``Cannot losslessly convert units``. Keeping
    every candle index as UTC nanoseconds avoids that version-dependent path.
    """
    out = pd.DatetimeIndex(index)
    if out.tz is None:
        out = out.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    try:
        return out.as_unit("ns")
    except (AttributeError, TypeError, ValueError):
        # Compatibility fallback for older pandas releases.
        return pd.DatetimeIndex(pd.to_datetime(out, utc=True))

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
    # Keep the same closed-candle contract as fetch_ohlcv(). The extra rows
    # requested above ensure that removing the forming candle does not leave
    # callers with fewer candles than requested.
    df = drop_unclosed(df, timeframe)
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
    out = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    out.index = normalize_utc_index(out.index)
    return out


def drop_unclosed(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Drop the last candle if it is still forming (its close time is in the
    future). Signals and backtests must only act on CLOSED candles, otherwise
    "repainting" makes live results look better than history.
    """
    tf_s = TIMEFRAME_SECONDS.get(timeframe)
    if tf_s is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = normalize_utc_index(df.index)
    else:
        normalized = normalize_utc_index(df.index)
        if not df.index.equals(normalized):
            df = df.copy()
            df.index = normalized
    last_close = int(df.index[-1].timestamp()) + tf_s
    if last_close > int(pd.Timestamp.now(tz="UTC").timestamp()):
        return df.iloc[:-1]
    return df


def price_change_pct_24h(df: pd.DataFrame,
                         now: pd.Timestamp | None = None) -> float | None:
    """Return the percentage change from the last candle at least 24h ago.

    ``searchsorted(target)`` returns the first candle *after* the target. That
    is not the candle that was available 24 hours ago and is especially wrong
    on the 1d timeframe. Use the last candle at or before the target instead.
    The optional ``now`` argument makes this calculation deterministic in
    unit tests.
    """
    if df is None or len(df) < 2:
        return None

    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    index = pd.DatetimeIndex(df.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")

    target = now - pd.Timedelta(hours=24)
    old_i = index.searchsorted(target, side="right") - 1
    last_i = index.searchsorted(now, side="right") - 1
    if old_i < 0 or last_i < 0 or old_i >= len(df) or last_i >= len(df):
        return None

    old_close = float(df["close"].iloc[old_i])
    last_close = float(df["close"].iloc[last_i])
    if old_close == 0:
        return None
    return (last_close / old_close - 1.0) * 100.0


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
                f"https://fapi.binance.com/futures/data/openInterestHist"
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

    price_24h = price_change_pct_24h(df)

    return {
        "funding_pct": funding * 100.0,
        "oi_chg_24h_pct": oi_chg,
        "price_chg_24h_pct": price_24h,
        "source": source,
    }
