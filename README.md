# 📊 Crypto Signal System — Consensus Engine with Regime, MTF & Sentiment Layers

A Python system that fetches live crypto candles from the **Binance public API**
(no API keys required), computes **12 core technical indicators**, adds live
**derivatives sentiment** (funding rate + open interest), and combines everything
into one weighted consensus signal — shaped by two accuracy layers:

- **Regime filter** (Choppiness Index): trend votes are muted in chop,
  oscillator votes are muted in trends
- **Multi-timeframe confluence**: a 4h signal that fights the daily trend is
  suppressed to NEUTRAL

Final output: **STRONG BUY · BUY · NEUTRAL · SELL · STRONG SELL** per coin,
with ATR-based stop-loss / take-profit levels, shaped by three accuracy
layers: **regime** (Choppiness), **candlestick-pattern entry filter** and
**multi-timeframe confluence** — each toggleable and A/B-testable.

Included:

- **CLI runner** (`main.py`) — terminal report, live-watch mode, JSON output
- **Streamlit dashboard** (`app.py`) — market board, candlestick charts,
  per-indicator vote tables, signal cards for 18 coins
- **Backtester** (`src/backtest.py`) — replays the *live engine* on history
- **Walk-forward validator** (`src/validation.py`) — out-of-sample testing +
  parameter-sensitivity heatmap + one-click **A/B** for the pattern filter
  (the guard against overfitting)
- **Pattern detector** (`src/patterns.py`) — 9 deterministic candlestick
  patterns used as an entry veto, not a vote
- **Indicator library** (`src/indicators.py`) — pure pandas/numpy, no TA-Lib

## The 14 voting components

| # | Component | Bullish 🟢 | Bearish 🔴 | Weight |
|---|-----------|------------|------------|--------|
| 1 | RSI (14) | < 30 oversold | > 70 overbought | 1.00 |
| 2 | MACD (12,26,9) | MACD above signal | MACD below signal | 1.50 |
| 3 | Bollinger (20, 2) | below lower band | above upper band | 1.00 |
| 4 | Stochastic (14, 3) | K < 20 | K > 80 | 1.00 |
| 5 | EMA 9/21 (+50/200) | EMA9 > EMA21 | EMA9 < EMA21 | 1.50 |
| 6 | Supertrend (10, 3) | price above line | price below line | 1.25 |
| 7 | ADX (14) | ADX ≥ 25 & +DI > −DI | ADX ≥ 25 & −DI > +DI | 1.25 |
| 8 | ATR (14) | — volatility only, never votes — | | 0.00 |
| 9 | OBV (vs 20-EMA) | above trend | below trend | 1.25 |
| 10 | MFI (14) | MFI < 20 | MFI > 80 | 1.00 |
| 11 | CCI (20) | < −100 | > +100 | 1.00 |
| 12 | ROC (10) | > +0.5% | < −0.5% | 1.00 |
| 13 | Funding rate (live) | ≤ −0.10% crowded shorts | ≥ +0.10% crowded longs | 1.00 |
| 14 | Open Interest Δ 24h (live) | price↑ OI↑ | price↓ OI↑ / price↓ OI↓ | 1.00 |

`score = Σ(vote × weight) ÷ Σ(weights of the votes that fired)` → **−1 … +1**

| Score | Final signal (also needs ≥ 3 agreeing votes) |
|-------|-----------------------------------------------|
| ≥ +0.50 | STRONG BUY |
| ≥ +0.25 | BUY |
| −0.25 … +0.25 | NEUTRAL |
| ≤ −0.25 | SELL |
| ≤ −0.50 | STRONG SELL |

### Layer 1 — Regime (Choppiness Index, period 14)

| CHOP | Regime | Effect |
|------|--------|--------|
| < 38.2 | TRENDING | oscillator weights ×0.4 |
| 38.2 – 61.8 | MIXED | — |
| > 61.8 | CHOPPY | trend weights ×0.4 |

### Layer 2 — Multi-timeframe confluence

Parent timeframe map: `15m→1h, 1h→4h, 4h→1d, 1d→1w`.
If the parent's EMA21/50 trend opposes the signal direction → **NEUTRAL**
(shown as "suppressed" in the output).

**Per-timeframe default** (`MTF_DEFAULT_BY_TF`, set by walk-forward
evidence across BTC/ETH/SOL/DOGE):

| Timeframe | MTF default | Evidence |
|-----------|-------------|----------|
| 15m / 1h | ON | marginal help (DOGE 1h: +0.4% OOS) |
| 4h | ON | wins 4/4 coins, up to +22.9% OOS (SOL) |
| 1d | OFF | weekly parent too slow — 4/4 coins worse with it (SOL: −30 vs +37 OOS) |

Override in the CLI with `--mtf` / `--no-mtf`, or in the dashboard
Backtest / Walk-forward tabs (Auto / On / Off).

### Layer — Candlestick pattern entry filter

Nine strict, deterministic patterns (bullish/bearish engulfing, hammer,
shooting star, doji, three white soldiers, three black crows, morning star,
evening star) detected on the **last closed candle** only.

Patterns are **not a 15th vote** — they are an entry-quality filter:
a fresh bearish pattern vetoes a new long entry when the filter is enabled,
a bullish pattern tags the live signal as "confirmed" (display only). The
**⚖️ A/B** button in the
Walk-forward tab runs the identical validation with the filter ON and OFF
and reports the OOS delta, so each coin/timeframe decides for itself whether
the filter earns its keep. Toggleable in the Backtest / Walk-forward tabs.

**A/B verdict from the initial evidence (5 cases, 4 folds, 0.1% fees):**

| Case | ON | OFF | Δ | Verdict |
|------|----|-----|---|---------|
| BTC 4h | +12.75% | +12.14% | +0.62 | patterns |
| SOL 4h | +32.55% | +32.55% | 0.00 | tie |
| DOGE 4h | +23.57% | +29.35% | −5.78 | no patterns |
| ETH 1d | −1.83% | +1.34% | −3.17 | no patterns |
| BTC 1d | −25.14% | −25.07% | −0.07 | tie |

Mixed evidence → the filter ships **default OFF**; enable it per coin where
the A/B test supports it.

### Derivatives sentiment (live only)

Funding rate + 24h open-interest change from the Binance futures public API
(OKX funding fallback when Binance is region-blocked). Backtests exclude this
layer (no free historical feed) — it can shift the score by at most 2 votes.

## Requirements

- Python **3.10+**
- Internet access (Binance public endpoints; automatic fallback to
  `data-api.binance.vision`, then OKX)

## Setup in PyCharm

1. **File → Open** → select this project folder.
2. Let PyCharm create a **Virtual Environment** (or: *Settings → Project →
   Python Interpreter → Add Interpreter → Virtualenv*).
3. In the PyCharm **Terminal** (bottom panel):
   ```
   pip install -r requirements.txt
   ```
4. Create run configurations (**Run → Edit Configurations → +**):
   - **Signal CLI** — type *Python*, script `main.py`,
     working directory = project root
   - **Dashboard** — type *Python*, module name `streamlit`,
     parameters `run app.py`, working directory = project root

   The dashboard can also be started by running `app.py` directly; it now
   forwards to Streamlit automatically instead of producing
   `missing ScriptRunContext` warnings. The module configuration above is
   still the recommended PyCharm setup.
5. Press **Run ▶**

## Usage

### CLI

```bash
python main.py                        # all 13 default coins, 4h candles
python main.py BTC/USDT ETH/USDT      # specific coins (any Binance spot pair)
python main.py --timeframe 1h         # other timeframes: 15m 1h 4h 1d
python main.py --no-mtf               # disable the higher-TF filter
python main.py --no-deriv             # disable funding/OI sentiment
python main.py --watch 300            # live mode: refresh every 5 minutes
python main.py --json                 # machine-readable output
```

### Dashboard

```bash
streamlit run app.py
```

Then open http://localhost:8501:

- **📊 Live signals** — market board (all coins at a glance) + per-coin
  sections: signal card (with regime, MTF and sentiment readout), chart with
  EMA/Supertrend/Bollinger overlays, full vote table
- **🧪 Backtest** — replay the live engine on up to 1000 candles with
  adjustable entry/exit thresholds and fees
- **🔬 Walk-forward** — out-of-sample validation + parameter-sensitivity
  heatmap
- **📒 Paper trades** — log the trades you actually act on in the local
  `paper_trades.json` journal, live P&L vs stop/target, position sizing for a
  1% risk, one-click close
- **ℹ️ How it works** — every rule, weight and threshold in one place
- **Execution guard** — on-demand candle freshness, public order-book spread/depth,
  VWAP fill estimate, risk budget, slippage, portfolio caps, and optional manual
  phone alert. It never places an order.

### A-grade quality gate

The mobile dashboard also includes a conservative meta-label for the current
signal. It replays historical signals with a triple-barrier label: the target
must be reached before the stop within the next few candles. If both levels are
inside the same candle, the stop is counted first. The gate reports the
Laplace-smoothed TP-first estimate, sample count, reward/risk, expected value
after estimated costs, signal expiry, and an A/B/WAIT grade.

This is an empirical filter, not a guarantee or a probability of profit. A
signal is marked **WAIT** when there are too few historical samples, the
conservative bound is weak, expected value is not positive, agreement is low,
or the regime/quality filters are unfavorable.

### Manual Binance Spot execution foundation

`src/broker.py`, `src/live_trading.py`, `src/trading_store.py`, and `worker.py`
provide a manual-approval/testnet-first execution foundation. Spot execution
accepts LONG plans only; SHORT plans require a separate futures adapter. The
worker reconciles protected orders and cancels a sibling exit after one closes.
Live submission is disabled unless all of these are deliberately configured in
Render: `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `BINANCE_SANDBOX`,
`TRADING_ENABLED`, `TRADING_MODE=manual`, and
`LIVE_TRADING_CONFIRMATION=I UNDERSTAND`. Never enable withdrawals on the API
key. Use a persistent disk or Postgres for `TRADING_DB_PATH` before trusting
live state across restarts.

## Configuration

Everything lives in **`src/config.py`**:

- coins — `MAJORS` (BTC, ETH, BNB, SOL, XRP), `ALT_MAJORS` (ADA, DOGE, AVAX,
  LINK, DOT, LTC, TRX, TON), `TRENDING` (SUI, APT, NEAR, AAVE, PEPE).
  `DEFAULT_COINS` = majors + alt majors (CLI default & dashboard pre-select);
  `ALL_COINS` is the full 18-coin list. Add any Binance spot pair.
- indicator periods & overbought/oversold levels (`IndicatorSettings`)
- consensus `WEIGHTS` per component
- signal thresholds (`STRONG_BUY`, `BUY`, `SELL`, `STRONG_SELL`)
- `MIN_CONFIRMATIONS` — agreeing votes needed for a directional signal
- `atr_stop_mult` / `atr_target_mult` — stop/target sizing
- regime cutoffs (`chop_trend`, `chop_choppy`, `regime_mute`)
- `PARENT_TIMEFRAME` — higher-timeframe map for the confluence filter
- `funding_hot_pct` / `oi_deadband_pct` — derivatives thresholds

## Project structure

```
crypto_signal_system/
├── main.py               # CLI runner (reports, live watch, JSON)
├── app.py                # Streamlit dashboard (signals, backtest,
│                         #   walk-forward validation, paper trades, docs)
├── paper_trades.example.json # empty journal template
├── paper_trades.json     # local journal, ignored by git
├── requirements.txt
├── README.md
└── src/
    ├── config.py         # coins, timeframes, weights, thresholds, layers
    ├── data.py           # Binance spot data + regional fallback chain +
    │                     #   funding/OI derivatives feed + closed-candle rule
    ├── indicators.py     # 12 indicators + Choppiness (pure pandas/numpy)
    ├── patterns.py       # 9 candlestick patterns (entry veto layer)
    ├── signals.py        # votes, regime weights, MTF + pattern filters, consensus
    ├── backtest.py       # backtest core (shared with validation)
    └── validation.py     # walk-forward + sensitivity grid + pattern A/B
```

## Correctness details that matter

- **Closed candles only** — the still-forming candle is dropped before
  signaling, and backtests only use parent-TF candles that were already
  closed at each signal bar (no look-ahead bias). Backtest signals are filled
  at the next candle open rather than the signal candle close.
- **Same engine everywhere** — live, backtest and validation share one code
  path (`row_votes` + `regime_mults` + `score_of`).
- **Graceful degradation** — if a derivatives feed is unreachable, those two
  votes simply don't fire and the denominator adjusts; the system never
  crashes on a missing feed.

## Extending

- **New coin** — add `"COIN/USDT"` to a group in `src/config.py`.
- **New indicator** — implement in `src/indicators.py`, add a vote in
  `src/signals.row_votes()`, register its weight in `src/config.WEIGHTS`,
  append a display row in `src/signals.evaluate()`.
- **Alerts** — inside the `--watch` loop in `main.py`, push `sig.final`
  changes to Telegram/Discord via a webhook (needs a bot token).
- **Deeper history** — `fetch_ohlcv_paged(symbol, timeframe, total)` in
  `src/data.py` pages past the 1000-candle cap via `startTime` (Binance),
  e.g. 4000×1h ≈ 6.6 months for long walk-forward runs.

## Disclaimer

Educational software. Technical indicators are **lagging**, derivatives
sentiment is a crowd gauge, and consensus is **not** a guarantee of profit —
fees, slippage and regime changes matter. Validate with the walk-forward tab
before believing any number. Nothing in this project is financial advice.

## Accuracy and risk enhancements

The research engine now includes several safeguards beyond adding more indicators:

- **Confidence panel** — the dashboard estimates historical results after 1, 3,
  and 5 completed intervals, including sample count, positive-result rate,
  average/median change, and efficiency. These are descriptive statistics,
  not guarantees.
- **Quality filters** — new directional statuses can be blocked when recent
  activity is too far below its rolling median or variation is outside the
  configured range (`min_volume_ratio`, `min_atr_pct`, `max_atr_pct`).
- **Realistic execution** — backtests use the next interval's open and include
  both processing cost and configurable execution variance/slippage.
- **Risk-based allocation** — simulations size each event from the ATR boundary,
  cap allocation, enforce a cooldown, and stop new entries after the daily loss
  limit is reached.
- **Robust validation** — rolling validation reports positive unseen sections,
  worst unseen section, average results, and sensitivity across thresholds.

Risk settings are in `IndicatorSettings`; the Scenario review and Rolling
validation tabs expose them without changing the underlying analysis rules.
