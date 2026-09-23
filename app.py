"""
Crypto Signal System — Streamlit dashboard.

Run from the project root:
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys

from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import indicators as ta
from src.backtest import run_backtest
from src.config import (ALL_COINS, ALT_MAJORS, DEFAULT_COINS, IndicatorSettings,
                        MAJORS, PARENT_TIMEFRAME, TIMEFRAMES, TRENDING)
from src.data import drop_unclosed, fetch_ohlcv
from src.signals import full_signal
from src.validation import sensitivity_grid, walk_forward

st.set_page_config(page_title="Crypto Signal System — 12-Indicator Consensus",
                   page_icon="📊", layout="wide")

CFG = IndicatorSettings()

SIGNAL_STYLE = {
    "STRONG BUY": ("#22c55e", "🟢🟢"),
    "BUY": ("#4ade80", "🟢"),
    "NEUTRAL": ("#94a3b8", "⚪"),
    "SELL": ("#fb923c", "🟠"),
    "STRONG SELL": ("#ef4444", "🔴🔴"),
}

MTF_LABEL = {1: "UP ✅", -1: "DOWN ⚠️", 0: "n/a"}


@st.cache_data(ttl=300, show_spinner="Fetching candles from Binance…")
def load_candles(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    return fetch_ohlcv(symbol, timeframe, limit)


@st.cache_data(ttl=300, show_spinner="Analyzing coin…")
def get_signal(symbol: str, timeframe: str, limit: int,
               use_mtf: bool = True, use_deriv: bool = True):
    return full_signal(symbol, timeframe, limit,
                       use_mtf=use_mtf, use_deriv=use_deriv)


@st.cache_data(ttl=300, show_spinner="Scanning the whole market board…")
def market_board(symbols: tuple, timeframe: str, limit: int) -> pd.DataFrame:
    """One row per coin: close, score, vote split, signal, MTF, ATR."""
    rows = []
    for s in symbols:
        try:
            sig = get_signal(s, timeframe, limit, use_mtf=None, use_deriv=False)
            pat = ""
            if sig.patterns and sig.patterns.names:
                pat = ("⛔ " if sig.patterns.bearish else
                       "✅ " if sig.patterns.bullish else "") + sig.patterns.names[0]
            rows.append({
                "Coin": s.replace("/USDT", ""),
                "Close (USDT)": f"{sig.close:,.6g}",
                "Score": round(sig.score, 2),
                "🟢 Bull": sig.bullish,
                "🔴 Bear": sig.bearish,
                "Signal": f"{SIGNAL_STYLE[sig.final][1]} {sig.final}",
                f"MTF {sig.mtf_timeframe or ''}".strip(): MTF_LABEL[sig.mtf_trend],
                "Regime": sig.regime,
                "Pattern": pat or "—",
                "ATR %": round(sig.atr_pct, 2),
            })
        except Exception:
            rows.append({"Coin": s.replace("/USDT", ""), "Close (USDT)": "—",
                         "Score": None, "🟢 Bull": 0, "🔴 Bear": 0,
                         "Signal": "⚠️ data error",
                         f"MTF {PARENT_TIMEFRAME.get(timeframe, '')}".strip(): "—",
                         "Regime": "—", "Pattern": "—", "ATR %": None})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- rendering
def price_chart(df: pd.DataFrame, ind: pd.DataFrame) -> go.Figure:
    view = df.tail(180)
    iv = ind.tail(180)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=view.index, open=view["open"], high=view["high"],
        low=view["low"], close=view["close"], name="Price"))
    for col, color, label in [
        ("ema_fast", "#38bdf8", "EMA 9"),
        ("ema_mid", "#f472b6", "EMA 21"),
        ("ema_trend_fast", "#facc15", "EMA 50"),
        ("ema_trend_slow", "#a78bfa", "EMA 200"),
        ("supertrend", "#2dd4bf", "Supertrend"),
    ]:
        fig.add_trace(go.Scatter(x=iv.index, y=iv[col], name=label,
                                 line=dict(width=1.1, color=color)))
    fig.add_trace(go.Scatter(x=iv.index, y=iv["bb_upper"], name="BB upper",
                             line=dict(width=0.8, color="rgba(148,163,184,0.55)")))
    fig.add_trace(go.Scatter(x=iv.index, y=iv["bb_lower"], name="BB lower",
                             line=dict(width=0.8, color="rgba(148,163,184,0.55)"),
                             fill="tonexty", fillcolor="rgba(148,163,184,0.07)"))
    fig.update_layout(height=540, template="plotly_dark",
                      xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
                      margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="Price (USDT)")
    return fig


def signal_card(sig) -> None:
    color, emoji = SIGNAL_STYLE[sig.final]
    sub = (f"consensus score <b>{sig.score:+.2f}</b>"
           f" &nbsp;·&nbsp; 🟢 {sig.bullish} &nbsp; 🔴 {sig.bearish} &nbsp; ⚪ {sig.neutral}"
           f" &nbsp;·&nbsp; agreement {sig.agreement}%")
    sub += (f"<br>regime <b>{sig.regime}</b> (CHOP {sig.chop:.0f})"
            f" &nbsp;·&nbsp; MTF {sig.mtf_timeframe or '—'}: {MTF_LABEL[sig.mtf_trend]}"
            + (" &nbsp;·&nbsp; <span style='color:#fb923c'>⚠ suppressed by higher-TF trend</span>"
               if sig.mtf_blocked else ""))
    if sig.deriv:
        fp = sig.deriv.get("funding_pct")
        oi = sig.deriv.get("oi_chg_24h_pct")
        d = []
        if fp is not None:
            d.append(f"funding {fp:+.4f}%")
        if oi is not None:
            d.append(f"OI 24h {oi:+.1f}%")
        if d:
            sub += f" &nbsp;·&nbsp; <span style='color:#7dd3fc'>{sig.deriv['source']}:" + " ".join(d) + "</span>"
    if sig.patterns:
        if sig.patterns.names:
            note = ", ".join(sig.patterns.names)
            if sig.patterns.bearish:
                sub += f" &nbsp;·&nbsp; <span style='color:#f87171'>⛔ pattern: {note} — long entry vetoed</span>"
            elif sig.patterns.bullish:
                sub += f" &nbsp;·&nbsp; <span style='color:#4ade80'>✅ pattern: {note} — entry confirmed</span>"
            else:
                sub += f" &nbsp;·&nbsp; pattern: {note} (indecision)"
        else:
            sub += " &nbsp;·&nbsp; pattern: none"
    st.markdown(
        f"""
        <div style="border:2px solid {color};background:{color}14;border-radius:14px;
                    padding:16px 24px;text-align:center;">
          <div style="font-size:1.9rem;font-weight:800;color:{color};">
            {emoji}&nbsp;&nbsp;{sig.final}
          </div>
          <div style="color:#94a3b8;font-size:0.85rem;margin-top:6px;">{sub}</div>
        </div>
        """, unsafe_allow_html=True)


def votes_table(sig) -> pd.DataFrame:
    rows = []
    for v in sig.votes:
        emoji = {1: "🟢 +1", 0: "⚪ 0", -1: "🔴 −1"}[v.vote]
        muted = " (muted by regime)" if v.vote == 0 and v.weight < 0.4 else ""
        rows.append({"Indicator": v.name, "Latest": v.value, "Vote": emoji,
                     "Weight": round(v.weight, 2), "Why": v.reason + muted})
    return pd.DataFrame(rows)


def coin_section(symbol: str, timeframe: str, limit: int) -> None:
    st.subheader(f"{symbol} — {timeframe}")
    try:
        sig = get_signal(symbol, timeframe, limit, use_mtf=None, use_deriv=True)
        df = load_candles(symbol, timeframe, limit)
        df = drop_unclosed(df, timeframe)
    except Exception as e:
        st.error(f"Could not load {symbol}: {e}")
        return

    ind = ta.compute_all(df, CFG)

    c1, c2, c3, c4 = st.columns([2.4, 1.1, 1.1, 1.1])
    with c1:
        signal_card(sig)
    with c2:
        st.metric("Close (USDT)", f"{sig.close:,.6g}",
                  f"{df['close'].pct_change().iloc[-1] * 100:+.2f}% last closed candle")
    with c3:
        st.metric("Consensus score", f"{sig.score:+.2f}", "range −1 … +1")
    with c4:
        st.metric("ATR volatility", f"{sig.atr_pct:.2f}%")

    lc, rc = st.columns(2)
    with lc:
        st.markdown("**Long setup (ATR-based levels)**")
        st.table(pd.DataFrame({
            "Level": ["Entry", "Stop-loss (1.5×ATR)", "Take-profit (2×ATR)"],
            "Price": [f"{sig.close:,.6g}",
                      f"{sig.stop_long:,.6g} ({(sig.stop_long / sig.close - 1) * 100:+.2f}%)",
                      f"{sig.target_long:,.6g} ({(sig.target_long / sig.close - 1) * 100:+.2f}%)"],
        }))
    with rc:
        st.markdown("**Short setup (ATR-based levels)**")
        st.table(pd.DataFrame({
            "Level": ["Entry", "Stop-loss (1.5×ATR)", "Take-profit (2×ATR)"],
            "Price": [f"{sig.close:,.6g}",
                      f"{sig.stop_short:,.6g} ({(sig.stop_short / sig.close - 1) * 100:+.2f}%)",
                      f"{sig.target_short:,.6g} ({(sig.target_short / sig.close - 1) * 100:+.2f}%)"],
        }))

    st.plotly_chart(price_chart(df, ind), use_container_width=True)
    st.markdown("**Indicator & sentiment votes (latest closed candle)**")
    st.dataframe(votes_table(sig), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------- tabs
def backtest_tab() -> None:
    st.markdown(
        "Replay the **live engine** (12 votes + regime filter + multi-timeframe filter) "
        "on historical candles: go **long** when the score crosses *above* the entry "
        "threshold and the higher timeframe is not pointing down, **exit** when it "
        "crosses *below* the exit threshold. Fees on both sides.")

    c1, c2, c3 = st.columns(3)
    with c1:
        coin = st.selectbox("Coin", ALL_COINS, key="bt_coin")
    with c2:
        tf = st.selectbox("Timeframe", TIMEFRAMES, index=2, key="bt_tf")
    with c3:
        limit = st.slider("Candles", 300, 1000, 1000, step=100, key="bt_limit")

    c4, c5, c6 = st.columns(3)
    with c4:
        entry_th = st.slider("Entry threshold (score)", -0.95, 0.95, 0.25, 0.05, key="bt_entry")
    with c5:
        exit_th = st.slider("Exit threshold (score)", -0.95, 0.95, -0.25, 0.05, key="bt_exit")
    with c6:
        fee = st.slider("Fee per side (%)", 0.0, 0.5, 0.1, 0.01, key="bt_fee")
    mtf_choice = st.selectbox(
        "MTF filter", ["Auto (per-timeframe default)", "On", "Off"],
        index=0, key="bt_mtf",
        help="Walk-forward evidence: MTF helps on 15m/1h/4h and hurts on 1d "
             "(the weekly parent trend is too slow). Auto = that default.")
    mtf_val = {"Auto (per-timeframe default)": None, "On": True, "Off": False}[mtf_choice]
    pat_choice = st.selectbox(
        "Pattern entry filter", ["Off (A/B default)", "On (veto bearish patterns)"],
        index=0, key="bt_pat",
        help="A fresh bearish candlestick pattern (engulfing, shooting star, …) "
             "on the last closed candle blocks a new long entry. A/B evidence: "
             "helps BTC 4h, hurts DOGE 4h — Off is the default.")
    pat_val = pat_choice.startswith("On")

    if st.button("▶ Run backtest", type="primary"):
        try:
            df = drop_unclosed(load_candles(coin, tf, limit), tf)
            parent_tf = PARENT_TIMEFRAME.get(tf, "")
            parent = (drop_unclosed(load_candles(coin, parent_tf, 500), parent_tf)
                      if parent_tf else None)
            with st.spinner("Running the consensus engine across history…"):
                res = run_backtest(df, CFG, entry_th, exit_th, fee,
                                   parent_df=parent, parent_timeframe=parent_tf,
                                   use_mtf=mtf_val, use_patterns=pat_val)
            st.session_state["bt"] = res
            st.session_state["bt_params"] = (coin, tf, entry_th, exit_th, fee,
                                             mtf_choice, pat_choice)
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            return

    if "bt" in st.session_state:
        res = st.session_state["bt"]
        coin, tf, entry_th, exit_th, fee, mtf_choice, pat_choice = \
            st.session_state["bt_params"]
        s = res["stats"]
        m = st.columns(7)
        m[0].metric("Strategy return", f"{s['total_return_pct']:+.2f}%")
        m[1].metric("Buy & hold", f"{s['buy_hold_pct']:+.2f}%")
        m[2].metric("Max drawdown", f"{s['max_drawdown_pct']:.2f}%")
        m[3].metric("Trades", str(s["num_trades"]))
        m[4].metric("Win rate", f"{s['win_rate_pct']:.0f}%")
        pf = s["profit_factor"]
        m[5].metric("Profit factor", "∞" if pf == float("inf") else f"{pf:.2f}")
        m[6].metric("Entries vetoed by pattern",
                    str(s.get("entries_blocked_by_pattern", 0)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["equity"].index, y=res["equity"] * 100,
                                 name="Consensus strategy",
                                 line=dict(width=2, color="#22c55e")))
        fig.add_trace(go.Scatter(x=res["buy_hold"].index, y=res["buy_hold"] * 100,
                                 name="Buy & hold",
                                 line=dict(width=1.2, dash="dot", color="#94a3b8")))
        fig.update_layout(height=430, template="plotly_dark",
                          title="Growth of 100 (fees included)",
                          yaxis_title="index (start = 100)",
                          margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        if not res["trades"].empty:
            st.markdown(f"**{coin} · {tf} · entry ≥ {entry_th:+.2f} · "
                        f"exit ≤ {exit_th:+.2f} · fee {fee:.2f}%/side · MTF {mtf_choice}**")
            st.dataframe(res["trades"], use_container_width=True, hide_index=True)
        else:
            st.info("No trades were triggered with these thresholds — try a lower entry threshold.")


def validation_tab() -> None:
    st.markdown(
        "**Walk-forward validation** — the honest test for any indicator system.\n\n"
        "The sample is split into consecutive folds. In each fold the engine picks the "
        "best entry/exit thresholds on the first 70% (in-sample), then trades **only the "
        "unseen last 30%** (out-of-sample). You can only believe the out-of-sample "
        "number. Below it, a sensitivity heatmap shows whether returns are a robust "
        "plateau or a single overfit spike.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        coin = st.selectbox("Coin", ALL_COINS, key="wf_coin")
    with c2:
        tf = st.selectbox("Timeframe", TIMEFRAMES, index=2, key="wf_tf")
    with c3:
        fee = st.slider("Fee per side (%)", 0.0, 0.5, 0.1, 0.01, key="wf_fee")
    with c4:
        n_folds = st.slider("Folds", 3, 6, 4, key="wf_folds")
    mtf_choice = st.selectbox(
        "MTF filter", ["Auto (per-timeframe default)", "On", "Off"],
        index=0, key="wf_mtf",
        help="Compare On vs Off to see whether the higher-TF filter earns "
             "its keep on this coin/timeframe.")
    mtf_val = {"Auto (per-timeframe default)": None, "On": True, "Off": False}[mtf_choice]
    pat_choice = st.selectbox(
        "Pattern entry filter", ["Off (A/B default)", "On (veto bearish patterns)"],
        index=0, key="wf_pat")
    pat_val = pat_choice.startswith("On")

    if st.button("🔬 Run walk-forward validation", type="primary"):
        try:
            df = drop_unclosed(load_candles(coin, tf, 1000), tf)
            parent_tf = PARENT_TIMEFRAME.get(tf, "")
            parent = (drop_unclosed(load_candles(coin, parent_tf, 500), parent_tf)
                      if parent_tf else None)
            with st.spinner("Optimizing & validating fold by fold (this takes a moment)…"):
                wf = walk_forward(df, CFG, parent_df=parent, parent_timeframe=parent_tf,
                                  fee_pct=fee, n_folds=n_folds, use_mtf=mtf_val,
                                  use_patterns=pat_val)
                grid = sensitivity_grid(df, CFG, parent_df=parent,
                                        parent_timeframe=parent_tf, fee_pct=fee,
                                        use_patterns=pat_val)
            st.session_state["wf"] = wf
            st.session_state["wf_grid"] = grid
            st.session_state["wf_params"] = (coin, tf, n_folds, fee, mtf_choice, pat_choice)
        except Exception as e:
            st.error(f"Validation failed: {e}")
            return

    if st.button("⚖️ A/B test: pattern filter ON vs OFF",
                 help="Runs the walk-forward twice — with and without the "
                      "candlestick-pattern entry veto — on identical folds. "
                      "The OOS delta is the evidence."):
        try:
            df = drop_unclosed(load_candles(coin, tf, 1000), tf)
            parent_tf = PARENT_TIMEFRAME.get(tf, "")
            parent = (drop_unclosed(load_candles(coin, parent_tf, 500), parent_tf)
                      if parent_tf else None)
            with st.spinner("Running both variants fold by fold (takes a moment or two)…"):
                wf_on = walk_forward(df, CFG, parent_df=parent,
                                     parent_timeframe=parent_tf, fee_pct=fee,
                                     n_folds=n_folds, use_mtf=mtf_val, use_patterns=True)
                wf_off = walk_forward(df, CFG, parent_df=parent,
                                      parent_timeframe=parent_tf, fee_pct=fee,
                                      n_folds=n_folds, use_mtf=mtf_val, use_patterns=False)
            st.session_state["wf_ab"] = (coin, tf, n_folds, fee, wf_on, wf_off)
        except Exception as e:
            st.error(f"A/B validation failed: {e}")
            return

    if "wf_ab" in st.session_state:
        a_coin, a_tf, a_folds, a_fee, wf_on, wf_off = st.session_state["wf_ab"]
        on, off = wf_on["oos_total_pct"], wf_off["oos_total_pct"]
        verdict = ("The pattern filter **earns its keep** here — it improved "
                   "out-of-sample results." if on >= off else
                   "The pattern filter does **not** help here — it suppressed "
                   "entries that would have worked.")
        st.markdown(f"#### ⚖️ A/B result — {a_coin} · {a_tf} · {a_folds} folds · "
                    f"fee {a_fee:.2f}%/side (out-of-sample only)")
        cmp = pd.DataFrame({
            "OOS total": [f"{on:+.2f}%", f"{off:+.2f}%"],
            "OOS by fold":
                [" / ".join(f"{x:+.1f}" for x in wf_on["folds"]["out-of-sample %"])
                 , " / ".join(f"{x:+.1f}" for x in wf_off["folds"]["out-of-sample %"])],
        }, index=["Patterns ON", "Patterns OFF"])
        st.table(cmp)
        st.markdown(f"**Δ = {on - off:+.2f}% OOS → {verdict}**")
        st.divider()

    if "wf" in st.session_state:
        wf = st.session_state["wf"]
        coin, tf, n_folds, fee, mtf_choice, pat_choice = st.session_state["wf_params"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Out-of-sample total (the real number)",
                  f"{wf['oos_total_pct']:+.2f}%")
        m2.metric("In-sample (cherry-picked, do not trust)", "see table ↓")
        m3.metric("Buy & hold, full sample", f"{wf['buy_hold_pct']:+.2f}%")

        st.markdown(f"**Walk-forward folds — {coin} · {tf} · {n_folds} folds · fee {fee:.2f}%/side**")
        st.dataframe(wf["folds"], use_container_width=True, hide_index=True)

        st.markdown("**Parameter sensitivity — total return % on the full sample**\n\n"
                    "Wide green plateau = robust settings. A lone bright spike = overfit.")
        grid = st.session_state["wf_grid"]
        fig = go.Figure(go.Heatmap(
            z=grid.values,
            x=list(grid.columns),
            y=list(grid.index),
            colorscale="RdYlGn",
            zmid=0.0,
            colorbar=dict(title="%"),
            texttemplate="%{z:.0f}%",
            hovertemplate="<b>%{y}</b><br>%{x}<br>return: %{z:.1f}%<extra></extra>",
        ))
        fig.update_layout(height=420, template="plotly_dark",
                          title=f"{coin} · {tf} · full sample",
                          xaxis_title="exit threshold", yaxis_title="entry threshold",
                          margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Derivatives sentiment (funding/OI) is live-only and excluded from "
            "validation — it can shift the score by at most 2 votes.")


# ---------------------------------------------------------- paper trades
PAPER_TRADES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trades.json")


def _load_paper_trades() -> list:
    try:
        with open(PAPER_TRADES_FILE) as f:
            return json.load(f).get("trades", [])
    except Exception:
        return []


def _save_paper_trades(trades: list) -> None:
    with open(PAPER_TRADES_FILE, "w") as f:
        json.dump({"trades": trades}, f, indent=2)


def _last_price(coin: str, timeframe: str):
    try:
        df = drop_unclosed(load_candles(coin, timeframe, 3), timeframe)
        return float(df["close"].iloc[-1])
    except Exception:
        return None


def _pnl_pct(entry: float, last: float, direction: str) -> float:
    return (entry / last - 1) * 100 if direction == "SHORT" else (last / entry - 1) * 100


def paper_tab() -> None:
    st.markdown(
        "Trades you've actually acted on, tracked against live prices — the loop that "
        "closes the gap between backtests and reality. Prices are cached for 5 minutes; "
        "hit **Refresh data** for fresh ticks.")
    trades = _load_paper_trades()
    if not trades:
        st.info("No paper trades logged yet.")
        return

    open_trades = [t for t in trades if t.get("status") == "OPEN"]
    closed_trades = [t for t in trades if t.get("status") != "OPEN"]

    if open_trades:
        st.markdown("#### 🟢 Open")
        rows = []
        for t in open_trades:
            last = _last_price(t["coin"], t["timeframe"])
            entry, stop, target = float(t["entry"]), float(t["stop"]), float(t["target"])
            direction = t.get("direction", "LONG")
            if last is None:
                rows.append({"Coin": t["coin"].replace("/USDT", ""), "TF": t["timeframe"],
                             "Dir": direction, "Entry": f"{entry:,.6g}", "Stop": f"{stop:,.6g}",
                             "Target": f"{target:,.6g}", "Last": "—", "uP&L": "—",
                             "to stop": "—", "to target": "—", "Status": "no data"})
                continue
            if direction == "SHORT":
                stop_hit, tgt_hit = last >= stop, last <= target
                d_stop, d_tgt = (last / stop - 1) * 100, (last / target - 1) * 100
            else:
                stop_hit, tgt_hit = last <= stop, last >= target
                d_stop, d_tgt = (stop / last - 1) * 100, (target / last - 1) * 100
            flag = ("⛔ STOP HIT — close it" if stop_hit else
                    "✅ TARGET HIT — take profit" if tgt_hit else "in play")
            rows.append({
                "Coin": t["coin"].replace("/USDT", ""),
                "TF": t["timeframe"],
                "Dir": direction,
                "Entry": f"{entry:,.6g}",
                "Stop": f"{stop:,.6g}",
                "Target": f"{target:,.6g}",
                "Last": f"{last:,.6g}",
                "uP&L": f"{_pnl_pct(entry, last, direction):+.2f}%",
                "to stop": f"{d_stop:+.2f}%",
                "to target": f"{d_tgt:+.2f}%",
                "Status": flag,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        for t in open_trades:
            entry, stop = float(t["entry"]), float(t["stop"])
            risk_pct = abs(entry - stop) / entry * 100
            notional = (1000 * 0.01) / (risk_pct / 100)
            label = t["coin"].replace("/USDT", "")
            st.caption(
                f"**{label}**: stop is {risk_pct:.2f}% from entry → risking 1% of a "
                f"$1,000 account ≈ ${notional:,.0f} notional (≈ {notional / entry:.2f} {label}).")

        for t in open_trades:
            c1, c2 = st.columns([1, 3])
            with c1:
                if st.button("Close at market", key=f"close_{t['id']}", use_container_width=True):
                    last = _last_price(t["coin"], t["timeframe"])
                    if last is not None:
                        t.update(status="CLOSED", exit_price=last,
                                 closed_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                                 exit_reason="manual close at market",
                                 pnl_pct=round(_pnl_pct(float(t["entry"]), last,
                                                        t.get("direction", "LONG")), 3))
                        _save_paper_trades(trades)
                        st.rerun()
                    else:
                        st.error(f"Could not fetch a price for {t['coin']} to close the trade.")
            with c2:
                st.caption(
                    f"{t['coin']} · {t['timeframe']} · opened {t.get('opened_at', '—')} · "
                    f"entry score {t.get('score', '—')} · {t.get('regime', '')} · "
                    f"MTF {t.get('mtf', '—')} · pattern: {t.get('pattern', '—')}")

    if closed_trades:
        st.markdown("#### 🏁 Closed")
        st.dataframe(pd.DataFrame([
            {
                "Coin": t["coin"].replace("/USDT", ""),
                "TF": t["timeframe"],
                "Dir": t.get("direction", "LONG"),
                "Entry": f"{float(t['entry']):,.6g}",
                "Exit": f"{float(t['exit_price']):,.6g}" if t.get("exit_price") is not None else "—",
                "P&L": f"{t['pnl_pct']:+.2f}%" if t.get("pnl_pct") is not None else "—",
                "Opened": t.get("opened_at", ""),
                "Closed": t.get("closed_at", ""),
                "Reason": t.get("exit_reason", ""),
            } for t in closed_trades
        ]), use_container_width=True, hide_index=True)

    st.caption(
        "Rules of the road: exit when the **last closed candle** closes through your "
        "stop or target — not on wicks. If a new bearish pattern vetoes the setup or "
        "the MTF flips, log the reason when you close. The closed-trade table at the "
        "end of the month is your real accuracy report.")


def about_tab() -> None:
    st.markdown(
        """
## How the consensus works

Every component votes on the **latest CLOSED candle**:
🟢 **+1** bullish · ⚪ **0** neutral · 🔴 **−1** bearish.

### Voting components (12 price/volume + 2 derivatives)
| # | Component | Bullish (+1) | Bearish (−1) | Weight |
|---|-----------|--------------|--------------|--------|
| 1 | RSI (14) | RSI < 30 (oversold) | RSI > 70 (overbought) | 1.00 |
| 2 | MACD (12,26,9) | MACD above signal | MACD below signal | 1.50 |
| 3 | Bollinger (20,2) | below lower band | above upper band | 1.00 |
| 4 | Stochastic (14,3) | K < 20 | K > 80 | 1.00 |
| 5 | EMA 9/21 (+50/200) | EMA9 > EMA21 | EMA9 < EMA21 | 1.50 |
| 6 | Supertrend (10,3) | price above the line | price below the line | 1.25 |
| 7 | ADX (14) | ADX ≥ 25 & +DI > −DI | ADX ≥ 25 & −DI > +DI | 1.25 |
| 8 | ATR (14) | — volatility gauge, never votes — | | 0.00 |
| 9 | OBV (vs 20-EMA) | above trend | below trend | 1.25 |
| 10 | MFI (14) | MFI < 20 | MFI > 80 | 1.00 |
| 11 | CCI (20) | CCI < −100 | CCI > +100 | 1.00 |
| 12 | ROC (10) | ROC > +0.5% | ROC < −0.5% | 1.00 |
| 13 | Funding rate (live) | ≤ −0.10% (crowded shorts) | ≥ +0.10% (crowded longs) | 1.00 |
| 14 | Open Interest Δ 24h (live) | price↑ OI↑ (new longs) | price↓ OI↑ or price↓ OI↓ | 1.00 |

### Layer 1 — Regime filter (Choppiness Index 14)
| CHOP | Regime | Effect |
|------|--------|--------|
| < 38.2 | TRENDING | oscillator votes (RSI, Stoch, CCI, BB, MFI) muted ×0.4 |
| 38.2 – 61.8 | MIXED | no muting |
| > 61.8 | CHOPPY | trend votes (MACD, EMA, Supertrend, ADX, ROC) muted ×0.4 |

### Layer 2 — Candlestick pattern entry filter
Strict deterministic patterns: bullish/bearish engulfing, hammer, shooting
star, doji, three white soldiers / three black crows, morning / evening star.
A fresh **bearish** pattern on the last closed candle **vets out** a new long
entry; a bullish pattern tags the signal "confirmed" (display). Patterns are
NOT a 15th vote — they are an entry-quality filter.

**A/B walk-forward verdict (5 cases, 4 folds, 0.1% fees):**
| Case | ON | OFF | Δ |
|------|----|-----|---|
| BTC 4h | +12.75% | +12.14% | +0.62 |
| SOL 4h | +32.55% | +32.55% | 0.00 |
| DOGE 4h | +23.57% | +29.35% | −5.78 |
| ETH 1d | −1.83% | +1.34% | −3.17 |
| BTC 1d | −25.14% | −25.07% | −0.07 |

Mixed → **default OFF**; flip it per coin via the ⚖️ A/B button in this tab.

### Layer 3 — Multi-timeframe confluence
A signal that fights the higher timeframe's EMA21/50 trend
(15m→1h, 1h→4h, 4h→1d, 1d→1w) is **suppressed to NEUTRAL**.

**Per-timeframe default (from walk-forward evidence):**
| Timeframe | MTF default | Why |
|-----------|-------------|-----|
| 15m / 1h | ON | marginal help (DOGE 1h: +0.4%) |
| 4h | ON | wins 4/4 coins, up to +22.9% OOS (SOL) |
| 1d | OFF | the weekly parent trend is too slow; the 1d edge comes from buying dips against it — the filter suppressed those best trades (4/4 coins worse with it) |

### Score → signal
`score = Σ(vote × weight) ÷ Σ(weights of the votes that fired)` → **−1 … +1**

| Score | Final signal (also needs ≥ 3 agreeing votes) |
|-------|----------------------------------------------|
| ≥ +0.50 | STRONG BUY |
| ≥ +0.25 | BUY |
| −0.25 … +0.25 | NEUTRAL |
| ≤ −0.25 | SELL |
| ≤ −0.50 | STRONG SELL |

ATR(14) sizes the suggested levels: **stop-loss = 1.5 × ATR**, **take-profit = 2 × ATR**.

### Trusting the results
The **Walk-forward** tab re-optimizes thresholds on unseen slices of history —
if the out-of-sample curve holds up, the logic has real structure. If it
collapses, treat the in-sample numbers as noise.

> ⚠️ Educational tool built on lagging indicators — **not financial advice**.
> Always backtest, walk-forward validate and paper-trade before risking capital.
        """
    )


def main() -> None:
    st.title("📊 Crypto Signal System — 12-Indicator Consensus + Sentiment")
    st.caption(
        "Live Binance candles (no API keys) · 12 price/volume indicators · funding & "
        "open-interest sentiment · regime (Choppiness) & multi-timeframe filters · "
        "**educational tool, not financial advice**"
    )

    with st.sidebar:
        st.header("⚙️ Controls")
        st.multiselect("Coins", ALL_COINS, default=DEFAULT_COINS, key="coin_select")
        st.markdown("**Quick select:**")
        b1, b2 = st.columns(2)
        if b1.button("🌐 All 18", use_container_width=True):
            st.session_state["coin_select"] = ALL_COINS
            st.rerun()
        if b2.button("👑 Majors", use_container_width=True):
            st.session_state["coin_select"] = MAJORS
            st.rerun()
        b3, b4 = st.columns(2)
        if b3.button("🪙 Alt majors", use_container_width=True):
            st.session_state["coin_select"] = ALT_MAJORS
            st.rerun()
        if b4.button("🔥 Trending", use_container_width=True):
            st.session_state["coin_select"] = TRENDING
            st.rerun()
        timeframe = st.selectbox("Timeframe", TIMEFRAMES, index=2)
        limit = st.slider("Candle limit", 300, 1000, 500, step=100)
        if st.button("🔄 Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.caption("Data is cached for 5 minutes. Use Refresh to force a re-fetch.")

    tab_sig, tab_bt, tab_wf, tab_paper, tab_about = st.tabs(
        ["📊 Live signals", "🧪 Backtest", "🔬 Walk-forward", "📒 Paper trades",
         "ℹ️ How it works"])

    with tab_sig:
        coins = st.session_state.get("coin_select", DEFAULT_COINS)
        if not coins:
            st.info("Pick at least one coin in the sidebar.")
        else:
            mtf_col = f"MTF {PARENT_TIMEFRAME.get(timeframe, '')}".strip()
            st.markdown(f"#### 🌐 Market board — {len(coins)} coins, {timeframe}")
            st.dataframe(market_board(tuple(coins), timeframe, limit),
                         use_container_width=True, hide_index=True)
            st.divider()
            for s in coins:
                coin_section(s, timeframe, limit)
                st.divider()

    with tab_bt:
        backtest_tab()

    with tab_wf:
        validation_tab()

    with tab_paper:
        paper_tab()

    with tab_about:
        about_tab()


if __name__ == "__main__":
    main()
