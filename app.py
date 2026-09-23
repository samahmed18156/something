"""
Operations Workspace — Streamlit dashboard.

Run from the project root:
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
import uuid

from dataclasses import replace
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _running_under_streamlit() -> bool:
    """Return True when this file is executing inside Streamlit's runner."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


# Running ``python app.py`` directly bypasses Streamlit's runtime and produces
# misleading missing-ScriptRunContext warnings. Launch Streamlit automatically
# for PyCharm users who configured this as a normal Python script. When
# Streamlit itself executes the file, a ScriptRunContext already exists and
# this block is skipped.
if __name__ == "__main__" and not _running_under_streamlit():
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__), *sys.argv[1:]]
    raise SystemExit(stcli.main())


from src import indicators as ta
from src.backtest import run_backtest
from src.config import (ALL_COINS, ALT_MAJORS, DEFAULT_COINS, IndicatorSettings,
                        MAJORS, PARENT_TIMEFRAME, TIMEFRAMES, TRENDING)
from src.data import drop_unclosed, fetch_ohlcv
from src.signals import full_signal
from src.validation import confidence_report, sensitivity_grid, walk_forward

st.set_page_config(page_title="Operations Workspace",
                   page_icon="📊", layout="wide")

CFG = IndicatorSettings()

# Workplace-friendly labels. The underlying symbols and strategy names remain
# unchanged so the analysis engine keeps working, but they are not exposed in
# the Streamlit UI.
ITEM_LABELS = {symbol: f"Item {i:02d}"
               for i, symbol in enumerate(ALL_COINS, start=1)}
LABEL_TO_SYMBOL = {label: symbol for symbol, label in ITEM_LABELS.items()}
ITEM_OPTIONS = [ITEM_LABELS[symbol] for symbol in ALL_COINS]
DEFAULT_ITEM_OPTIONS = [ITEM_LABELS[symbol] for symbol in DEFAULT_COINS]
GROUP_OPTIONS = {
    "All items": [ITEM_LABELS[symbol] for symbol in ALL_COINS],
    "Primary group": [ITEM_LABELS[symbol] for symbol in MAJORS],
    "Secondary group": [ITEM_LABELS[symbol] for symbol in ALT_MAJORS],
    "Recent group": [ITEM_LABELS[symbol] for symbol in TRENDING],
}

STATUS_LABEL = {
    "STRONG BUY": "Priority Up",
    "BUY": "Upward",
    "NEUTRAL": "Stable",
    "SELL": "Downward",
    "STRONG SELL": "Priority Down",
}

SIGNAL_STYLE = {
    "STRONG BUY": ("#22c55e", "🟢🟢"),
    "BUY": ("#4ade80", "🟢"),
    "NEUTRAL": ("#94a3b8", "⚪"),
    "SELL": ("#fb923c", "🟠"),
    "STRONG SELL": ("#ef4444", "🔴🔴"),
}

MTF_LABEL = {1: "Aligned ✅", -1: "Opposed ⚠️", 0: "Unavailable"}


def display_item(symbol: str) -> str:
    return ITEM_LABELS.get(symbol, "Item")


def display_direction(direction: str) -> str:
    return "Positive" if direction == "LONG" else "Negative"


@st.cache_data(ttl=300, show_spinner="Loading current data…")
def load_candles(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    return fetch_ohlcv(symbol, timeframe, limit)


@st.cache_data(ttl=300, show_spinner="Analyzing item…")
def get_signal(symbol: str, timeframe: str, limit: int,
               use_mtf: bool | None = None, use_deriv: bool = True,
               use_patterns: bool = False):
    return full_signal(symbol, timeframe, limit,
                       use_mtf=use_mtf, use_deriv=use_deriv,
                       use_patterns=use_patterns)


@st.cache_data(ttl=300, show_spinner="Scanning items…")
def market_board(symbols: tuple, timeframe: str, limit: int,
                 use_patterns: bool = False) -> pd.DataFrame:
    """One row per item with neutral workplace-friendly labels."""
    rows = []
    for s in symbols:
        try:
            sig = get_signal(s, timeframe, limit, use_mtf=None, use_deriv=False,
                             use_patterns=use_patterns)
            pat = ""
            if sig.patterns and sig.patterns.names:
                pat_prefix = ("⛔ " if sig.pattern_vetoed else
                              "⚠ " if sig.patterns.bearish else
                              "✅ " if sig.patterns.bullish else "")
                pat = pat_prefix + sig.patterns.names[0]
            rows.append({
                "Item": display_item(s),
                "Current value": f"{sig.close:,.6g}",
                "Score": round(sig.score, 2),
                "Positive factors": sig.bullish,
                "Negative factors": sig.bearish,
                "Status": f"{SIGNAL_STYLE[sig.final][1]} {STATUS_LABEL[sig.final]}",
                "Secondary trend": MTF_LABEL[sig.mtf_trend],
                "Environment": sig.regime,
                "Event": "Detected" if sig.patterns and sig.patterns.names else "—",
                "Filter": "Active" if (sig.pattern_vetoed or sig.filter_blocked) else "—",
                "Variation": round(sig.atr_pct, 2),
            })
        except Exception:
            rows.append({"Item": display_item(s), "Current value": "—",
                         "Score": None, "Positive factors": 0, "Negative factors": 0,
                         "Status": "⚠️ unavailable", "Secondary trend": "—",
                         "Environment": "—", "Event": "—", "Filter": "—",
                         "Variation": None})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- rendering
def price_chart(df: pd.DataFrame, ind: pd.DataFrame) -> go.Figure:
    """Neutral line chart without finance-specific overlays."""
    view = df.tail(180)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view.index, y=view["close"], name="Trend",
        line=dict(width=2, color="#38bdf8"),
        mode="lines"))
    fig.update_layout(
        height=430, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Value",
        title="Trend overview",
    )
    return fig


def signal_card(sig) -> None:
    color, emoji = SIGNAL_STYLE[sig.final]
    status = STATUS_LABEL[sig.final]
    sub = (f"status <b>{status}</b> &nbsp;·&nbsp; score <b>{sig.score:+.2f}</b>"
           f" &nbsp;·&nbsp; positive {sig.bullish} &nbsp; negative {sig.bearish}"
           f" &nbsp;·&nbsp; consistency {sig.agreement}%")
    sub += (f"<br>environment <b>{sig.regime}</b> &nbsp;·&nbsp; "
            f"secondary trend: {MTF_LABEL[sig.mtf_trend]}"
            + (" &nbsp;·&nbsp; <span style='color:#fb923c'>⚠ secondary trend differs</span>"
               if sig.mtf_blocked else ""))
    if sig.filter_blocked:
        sub += " &nbsp;·&nbsp; <span style='color:#fbbf24'>⚠ quality gate active</span>"
    if sig.patterns and sig.patterns.names:
        if sig.pattern_vetoed:
            sub += " &nbsp;·&nbsp; <span style='color:#f87171'>⛔ reversal filter active</span>"
        else:
            sub += " &nbsp;·&nbsp; event flag detected"
    st.markdown(
        f"""
        <div style="border:2px solid {color};background:{color}14;border-radius:14px;
                    padding:16px 24px;text-align:center;">
          <div style="font-size:1.9rem;font-weight:800;color:{color};">
            {emoji}&nbsp;&nbsp;{status}
          </div>
          <div style="color:#94a3b8;font-size:0.85rem;margin-top:6px;">{sub}</div>
        </div>
        """, unsafe_allow_html=True)


def votes_table(sig) -> pd.DataFrame:
    """Show factor states without exposing indicator names."""
    rows = []
    for i, v in enumerate(sig.votes, start=1):
        state = {1: "Positive", 0: "Neutral", -1: "Negative"}[v.vote]
        rows.append({
            "Component": f"Factor {i:02d}",
            "Current state": state,
            "Importance": round(v.weight, 2),
            "Assessment": ("Favorable" if v.vote == 1 else
                           "Unfavorable" if v.vote == -1 else "No clear direction"),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner="Building historical confidence panel…")
def get_confidence(symbol: str, timeframe: str, limit: int,
                   use_patterns: bool = False, slippage_pct: float = 0.05):
    df = drop_unclosed(load_candles(symbol, timeframe, limit), timeframe)
    parent_tf = PARENT_TIMEFRAME.get(timeframe, "")
    parent = (drop_unclosed(load_candles(symbol, parent_tf, 500), parent_tf)
              if parent_tf else None)
    return confidence_report(
        df, parent_df=parent, parent_timeframe=parent_tf, timeframe=timeframe,
        use_mtf=None, use_patterns=use_patterns, slippage_pct=slippage_pct)


def _format_pf(value: float) -> str:
    return "∞" if value == float("inf") else f"{value:.2f}"


def coin_section(symbol: str, timeframe: str, limit: int,
                 use_patterns: bool = False) -> None:
    st.subheader(f"{display_item(symbol)} — {timeframe}")
    try:
        sig = get_signal(symbol, timeframe, limit, use_mtf=None, use_deriv=True,
                         use_patterns=use_patterns)
        df = load_candles(symbol, timeframe, limit)
        df = drop_unclosed(df, timeframe)
    except Exception as e:
        st.error(f"Could not load {display_item(symbol)}: {e}")
        return

    ind = ta.compute_all(df, CFG)

    c1, c2, c3, c4 = st.columns([2.4, 1.1, 1.1, 1.1])
    with c1:
        signal_card(sig)
    with c2:
        st.metric("Current value", f"{sig.close:,.6g}",
                  f"{df['close'].pct_change().iloc[-1] * 100:+.2f}% latest interval")
    with c3:
        st.metric("Consistency score", f"{sig.score:+.2f}", "range −1 … +1")
    with c4:
        st.metric("Variation", f"{sig.atr_pct:.2f}%")

    lc, rc = st.columns(2)
    with lc:
        st.markdown("**Positive scenario**")
        st.table(pd.DataFrame({
            "Level": ["Reference", "Lower boundary", "Upper objective"],
            "Value": [f"{sig.close:,.6g}",
                      f"{sig.stop_long:,.6g} ({(sig.stop_long / sig.close - 1) * 100:+.2f}%)",
                      f"{sig.target_long:,.6g} ({(sig.target_long / sig.close - 1) * 100:+.2f}%)"],
        }))
    with rc:
        st.markdown("**Negative scenario**")
        st.table(pd.DataFrame({
            "Level": ["Reference", "Upper boundary", "Lower objective"],
            "Value": [f"{sig.close:,.6g}",
                      f"{sig.stop_short:,.6g} ({(sig.stop_short / sig.close - 1) * 100:+.2f}%)",
                      f"{sig.target_short:,.6g} ({(sig.target_short / sig.close - 1) * 100:+.2f}%)"],
        }))

    st.plotly_chart(price_chart(df, ind), use_container_width=True)
    st.markdown("**Component assessment for the latest interval**")
    st.dataframe(votes_table(sig), use_container_width=True, hide_index=True)

    with st.expander("Confidence and historical consistency"):
        try:
            report = get_confidence(symbol, timeframe, limit, use_patterns)
            current = report["current"]
            st.caption(
                f"Current status: **{current['status']}** · score {current['score']:+.2f}. "
                "Historical results are descriptive and are not guarantees.")
            rows = []
            for horizon, stats in report["horizons"].items():
                rows.append({
                    "Forward interval": horizon,
                    "Samples": stats["samples"],
                    "Positive result": f"{stats['win_rate_pct']:.1f}%",
                    "Average change": f"{stats['average_return_pct']:+.2f}%",
                    "Median change": f"{stats['median_return_pct']:+.2f}%",
                    "Efficiency": _format_pf(stats["profit_factor"]),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            direction_rows = []
            for label, key in (("Positive", "positive"), ("Negative", "negative")):
                stats = report["directions"][key]
                direction_rows.append({
                    "Scenario": label, "Samples": stats["samples"],
                    "Positive result": f"{stats['win_rate_pct']:.1f}%",
                    "Average change": f"{stats['average_return_pct']:+.2f}%",
                })
            st.dataframe(pd.DataFrame(direction_rows), use_container_width=True, hide_index=True)
            if current["filter_reasons"]:
                st.info("Current quality gates: " + ", ".join(current["filter_reasons"]))
        except Exception as e:
            st.info(f"Confidence panel unavailable: {e}")

    with st.expander("Risk controls"):
        stop_distance = abs(sig.stop_long / sig.close - 1.0) * 100.0
        allocation = (min(CFG.max_position_pct,
                          CFG.risk_per_trade_pct / stop_distance * 100.0)
                      if stop_distance > 0 else CFG.max_position_pct)
        st.dataframe(pd.DataFrame([{
            "Risk per event": f"{CFG.risk_per_trade_pct:.2f}%",
            "Maximum allocation": f"{CFG.max_position_pct:.0f}%",
            "Estimated allocation": f"{allocation:.1f}%",
            "Daily loss cap": f"{CFG.max_daily_loss_pct:.2f}%",
            "Cooldown": f"{CFG.signal_cooldown_bars} intervals",
        }]), use_container_width=True, hide_index=True)


# ----------------------------------------------------------------- tabs
def backtest_tab() -> None:
    st.markdown(
        "Replay the analysis engine over historical data. The positive and negative "
        "thresholds define two possible scenarios; processing costs are included.")

    c1, c2, c3 = st.columns(3)
    with c1:
        item_label = st.selectbox("Item", ITEM_OPTIONS, key="bt_item")
        coin = LABEL_TO_SYMBOL[item_label]
    with c2:
        tf = st.selectbox("Interval", TIMEFRAMES, index=2, key="bt_tf")
    with c3:
        limit = st.slider("Data points", 300, 1000, 1000, step=100, key="bt_limit")

    c4, c5, c6 = st.columns(3)
    with c4:
        entry_th = st.slider("Positive threshold", -0.95, 0.95, 0.25, 0.05, key="bt_entry")
    with c5:
        exit_th = st.slider("Negative threshold", -0.95, 0.95, -0.25, 0.05, key="bt_exit")
    with c6:
        fee = st.slider("Processing cost (%)", 0.0, 0.5, 0.1, 0.01, key="bt_fee")
    slippage = st.slider("Execution variance per side (%)", 0.0, 0.5, 0.05, 0.01, key="bt_slippage")

    r1, r2, r3 = st.columns(3)
    with r1:
        risk_pct = st.slider("Risk per event (%)", 0.1, 5.0, float(CFG.risk_per_trade_pct),
                             0.1, key="bt_risk")
    with r2:
        max_position = st.slider("Maximum allocation (%)", 5.0, 100.0,
                                 float(CFG.max_position_pct), 5.0, key="bt_max_position")
    with r3:
        daily_loss = st.slider("Daily loss cap (%)", 0.5, 10.0,
                               float(CFG.max_daily_loss_pct), 0.5, key="bt_daily_loss")

    alignment_choice = st.selectbox(
        "Secondary alignment", ["Auto (standard)", "Aligned only", "All data"],
        index=0, key="bt_mtf")
    mtf_val = {"Auto (standard)": None, "Aligned only": True,
               "All data": False}[alignment_choice]
    filter_choice = st.selectbox(
        "Reversal filter", ["Off (default)", "On"],
        index=0, key="bt_pat",
        help="When enabled, a fresh reversal event can block a positive scenario.")
    filter_val = filter_choice == "On"

    if st.button("▶ Run scenario review", type="primary"):
        try:
            df = drop_unclosed(load_candles(coin, tf, limit), tf)
            parent_tf = PARENT_TIMEFRAME.get(tf, "")
            parent = (drop_unclosed(load_candles(coin, parent_tf, 500), parent_tf)
                      if parent_tf else None)
            sim_cfg = replace(
                CFG, risk_per_trade_pct=risk_pct,
                max_position_pct=max_position, max_daily_loss_pct=daily_loss)
            with st.spinner("Processing historical data…"):
                res = run_backtest(
                    df, sim_cfg, entry_th, exit_th, fee,
                    parent_df=parent, parent_timeframe=parent_tf,
                    use_mtf=mtf_val, use_patterns=filter_val, timeframe=tf,
                    slippage_pct=slippage)
            st.session_state["bt"] = res
            st.session_state["bt_params"] = (item_label, tf, entry_th, exit_th, fee,
                                               slippage, risk_pct, max_position, daily_loss,
                                               alignment_choice, filter_choice)
        except Exception as e:
            st.error(f"Scenario review failed: {e}")
            return

    if "bt" in st.session_state:
        res = st.session_state["bt"]
        params = st.session_state["bt_params"]
        if len(params) == 7:  # compatibility with an already-open old session
            item_label, tf, entry_th, exit_th, fee, alignment_choice, filter_choice = params
            slippage, risk_pct, max_position, daily_loss = 0.05, CFG.risk_per_trade_pct, CFG.max_position_pct, CFG.max_daily_loss_pct
        else:
            (item_label, tf, entry_th, exit_th, fee, slippage, risk_pct,
             max_position, daily_loss, alignment_choice, filter_choice) = params
        stats = res["stats"]
        m = st.columns(9)
        m[0].metric("Scenario result", f"{stats['total_return_pct']:+.2f}%")
        m[1].metric("Baseline", f"{stats['buy_hold_pct']:+.2f}%")
        m[2].metric("Largest decline", f"{stats['max_drawdown_pct']:.2f}%")
        m[3].metric("Events", str(stats["num_trades"]))
        m[4].metric("Positive ratio", f"{stats['win_rate_pct']:.0f}%")
        pf = stats["profit_factor"]
        m[5].metric("Efficiency", "∞" if pf == float("inf") else f"{pf:.2f}")
        m[6].metric("Filtered events", str(stats.get("entries_blocked_by_pattern", 0) + stats.get("entries_blocked_by_quality", 0)))
        m[7].metric("Sharpe", f"{stats.get('sharpe', 0.0):.2f}")
        m[8].metric("Avg allocation", f"{stats.get('average_position_pct', 0.0):.1f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res["equity"].index, y=res["equity"] * 100,
                                 name="Scenario", line=dict(width=2, color="#22c55e")))
        fig.add_trace(go.Scatter(x=res["buy_hold"].index, y=res["buy_hold"] * 100,
                                 name="Baseline", line=dict(width=1.2, dash="dot",
                                                              color="#94a3b8")))
        fig.update_layout(height=430, template="plotly_dark",
                          title="Indexed result (start = 100)",
                          yaxis_title="Index", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        if not res["trades"].empty:
            st.markdown(f"**{item_label} · {tf} · positive ≥ {entry_th:+.2f} · "
                        f"negative ≤ {exit_th:+.2f} · processing {fee:.2f}% · variance {slippage:.2f}%**")
            shown = res["trades"].rename(columns={
                "entry_time": "Start time", "exit_time": "End time",
                "entry": "Start value", "exit": "End value",
                "pnl_pct": "Change %", "account_pnl_pct": "Allocation change %",
                "position_pct": "Allocation %", "bars_held": "Intervals",
            })
            st.dataframe(shown, use_container_width=True, hide_index=True)
        else:
            st.info("No events matched these thresholds. Try a lower positive threshold.")

def validation_tab() -> None:
    st.markdown(
        "**Rolling validation** — settings are selected on an earlier slice of the "
        "data and then evaluated only on the later unseen slice. The unseen result "
        "is the number to focus on.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        item_label = st.selectbox("Item", ITEM_OPTIONS, key="wf_item")
        coin = LABEL_TO_SYMBOL[item_label]
    with c2:
        tf = st.selectbox("Interval", TIMEFRAMES, index=2, key="wf_tf")
    with c3:
        fee = st.slider("Processing cost (%)", 0.0, 0.5, 0.1, 0.01, key="wf_fee")
    with c4:
        n_folds = st.slider("Validation sections", 3, 6, 4, key="wf_folds")
    slippage = st.slider("Execution variance per side (%)", 0.0, 0.5, 0.05, 0.01, key="wf_slippage")
    vr1, vr2, vr3 = st.columns(3)
    with vr1:
        risk_pct = st.slider("Risk per event (%)", 0.1, 5.0, float(CFG.risk_per_trade_pct),
                             0.1, key="wf_risk")
    with vr2:
        max_position = st.slider("Maximum allocation (%)", 5.0, 100.0,
                                 float(CFG.max_position_pct), 5.0, key="wf_max_position")
    with vr3:
        daily_loss = st.slider("Daily loss cap (%)", 0.5, 10.0,
                               float(CFG.max_daily_loss_pct), 0.5, key="wf_daily_loss")

    alignment_choice = st.selectbox(
        "Secondary alignment", ["Auto (standard)", "Aligned only", "All data"],
        index=0, key="wf_mtf")
    mtf_val = {"Auto (standard)": None, "Aligned only": True,
               "All data": False}[alignment_choice]
    filter_choice = st.selectbox(
        "Reversal filter", ["Off (default)", "On"], index=0, key="wf_pat")
    filter_val = filter_choice == "On"

    if st.button("🔬 Run rolling validation", type="primary"):
        try:
            df = drop_unclosed(load_candles(coin, tf, 1000), tf)
            parent_tf = PARENT_TIMEFRAME.get(tf, "")
            parent = (drop_unclosed(load_candles(coin, parent_tf, 500), parent_tf)
                      if parent_tf else None)
            with st.spinner("Evaluating section by section…"):
                validation_cfg = replace(
                    CFG, risk_per_trade_pct=risk_pct,
                    max_position_pct=max_position, max_daily_loss_pct=daily_loss)
                wf = walk_forward(
                    df, validation_cfg, parent_df=parent, parent_timeframe=parent_tf,
                    fee_pct=fee, n_folds=n_folds, use_mtf=mtf_val,
                    use_patterns=filter_val, timeframe=tf, slippage_pct=slippage)
                grid = sensitivity_grid(
                    df, validation_cfg, parent_df=parent, parent_timeframe=parent_tf,
                    fee_pct=fee, use_patterns=filter_val, use_mtf=mtf_val,
                    timeframe=tf, slippage_pct=slippage)
            st.session_state["wf"] = wf
            st.session_state["wf_grid"] = grid
            st.session_state["wf_params"] = (item_label, tf, n_folds, fee,
                                               alignment_choice, filter_choice)
        except Exception as e:
            st.error(f"Validation failed: {e}")
            return

    if st.button("⚖️ Compare reversal filter",
                 help="Runs the same rolling validation with the filter enabled and disabled."):
        try:
            df = drop_unclosed(load_candles(coin, tf, 1000), tf)
            parent_tf = PARENT_TIMEFRAME.get(tf, "")
            parent = (drop_unclosed(load_candles(coin, parent_tf, 500), parent_tf)
                      if parent_tf else None)
            validation_cfg = replace(
                CFG, risk_per_trade_pct=risk_pct,
                max_position_pct=max_position, max_daily_loss_pct=daily_loss)
            with st.spinner("Comparing both configurations…"):
                wf_on = walk_forward(
                    df, validation_cfg, parent_df=parent, parent_timeframe=parent_tf,
                    fee_pct=fee, n_folds=n_folds, use_mtf=mtf_val,
                    use_patterns=True, timeframe=tf, slippage_pct=slippage)
                wf_off = walk_forward(
                    df, validation_cfg, parent_df=parent, parent_timeframe=parent_tf,
                    fee_pct=fee, n_folds=n_folds, use_mtf=mtf_val,
                    use_patterns=False, timeframe=tf, slippage_pct=slippage)
            st.session_state["wf_ab"] = (item_label, tf, n_folds, fee, wf_on, wf_off)
        except Exception as e:
            st.error(f"Comparison failed: {e}")
            return

    if "wf_ab" in st.session_state:
        a_item, a_tf, a_folds, a_fee, wf_on, wf_off = st.session_state["wf_ab"]
        on, off = wf_on["oos_total_pct"], wf_off["oos_total_pct"]
        verdict = ("The enabled filter improved the unseen result." if on >= off
                   else "The disabled filter performed better on the unseen result.")
        st.markdown(f"#### ⚖️ Configuration comparison — {a_item} · {a_tf} · "
                    f"{a_folds} sections · processing {a_fee:.2f}%")
        cmp = pd.DataFrame({
            "Unseen total": [f"{on:+.2f}%", f"{off:+.2f}%"],
            "By section": [
                " / ".join(f"{x:+.1f}" for x in wf_on["folds"]["out-of-sample %"]),
                " / ".join(f"{x:+.1f}" for x in wf_off["folds"]["out-of-sample %"]),
            ],
        }, index=["Filter enabled", "Filter disabled"])
        st.table(cmp)
        st.markdown(f"**Difference = {on - off:+.2f}% → {verdict}**")
        st.divider()

    if "wf" in st.session_state:
        wf = st.session_state["wf"]
        item_label, tf, n_folds, fee, alignment_choice, filter_choice = \
            st.session_state["wf_params"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Unseen total", f"{wf['oos_total_pct']:+.2f}%")
        m2.metric("Positive sections", f"{wf.get('oos_positive_folds', 0)}/{wf.get('oos_fold_count', 0)}")
        m3.metric("Worst section", f"{wf.get('oos_worst_fold_pct', 0.0):+.2f}%")
        m4.metric("Baseline, full data", f"{wf['buy_hold_pct']:+.2f}%")
        st.markdown(f"**Validation sections — {item_label} · {tf} · {n_folds} sections · "
                    f"processing {fee:.2f}%**")
        st.dataframe(wf["folds"], use_container_width=True, hide_index=True)

        st.markdown("**Sensitivity overview — full data**")
        grid = st.session_state["wf_grid"]
        fig = go.Figure(go.Heatmap(
            z=grid.values, x=list(grid.columns), y=list(grid.index),
            colorscale="RdYlGn", zmid=0.0, colorbar=dict(title="%"),
            texttemplate="%{z:.0f}%",
            hovertemplate="<b>%{y}</b><br>%{x}<br>result: %{z:.1f}%<extra></extra>",
        ))
        fig.update_layout(height=420, template="plotly_dark",
                          title=f"{item_label} · {tf} · full data",
                          xaxis_title="Lower threshold", yaxis_title="Upper threshold",
                          margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)


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
        "Records you choose to monitor, tracked against the latest available values. "
        "All records are stored locally in this workspace.")
    records = _load_paper_trades()

    with st.expander("➕ Add activity record", expanded=not bool(records)):
        with st.form("new_activity_record"):
            c1, c2, c3 = st.columns(3)
            with c1:
                new_item_label = st.selectbox("Item", ITEM_OPTIONS, key="paper_new_item")
            with c2:
                new_tf = st.selectbox("Interval", TIMEFRAMES, index=2, key="paper_new_tf")
            with c3:
                scenario_label = st.selectbox("Scenario", ["Positive", "Negative"],
                                              key="paper_new_direction")
            c4, c5, c6 = st.columns(3)
            with c4:
                new_entry = st.number_input("Reference value", min_value=0.0, value=0.0,
                                            format="%.8f", key="paper_new_entry")
            with c5:
                new_stop = st.number_input("Lower/upper boundary", min_value=0.0, value=0.0,
                                           format="%.8f", key="paper_new_stop")
            with c6:
                new_target = st.number_input("Objective value", min_value=0.0, value=0.0,
                                             format="%.8f", key="paper_new_target")
            new_note = st.text_input("Note", value="", key="paper_new_note")
            submitted = st.form_submit_button("Save activity record", type="primary")

        if submitted:
            new_direction = "LONG" if scenario_label == "Positive" else "SHORT"
            valid = new_entry > 0 and new_stop > 0 and new_target > 0
            if new_direction == "LONG":
                valid = valid and new_stop < new_entry < new_target
            else:
                valid = valid and new_target < new_entry < new_stop
            if not valid:
                st.error("Use positive values with Positive: boundary < reference < objective, "
                         "or Negative: objective < reference < boundary.")
            else:
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                records.append({
                    "id": uuid.uuid4().hex[:12],
                    "coin": LABEL_TO_SYMBOL[new_item_label],
                    "timeframe": new_tf,
                    "direction": new_direction,
                    "status": "OPEN",
                    "opened_at": now,
                    "entry": float(new_entry),
                    "stop": float(new_stop),
                    "target": float(new_target),
                    "score": None,
                    "agreement_pct": None,
                    "regime": "manual record",
                    "mtf": "—",
                    "pattern": new_note or "—",
                    "closed_at": None,
                    "exit_price": None,
                    "exit_reason": None,
                    "pnl_pct": None,
                })
                _save_paper_trades(records)
                st.success("Activity record saved.")
                st.rerun()

    if not records:
        st.info("No activity records yet.")
        return

    active = [t for t in records if t.get("status") == "OPEN"]
    completed = [t for t in records if t.get("status") != "OPEN"]

    if active:
        st.markdown("#### 🟢 Active")
        rows = []
        for t in active:
            last = _last_price(t["coin"], t["timeframe"])
            entry, stop, target = float(t["entry"]), float(t["stop"]), float(t["target"])
            direction = t.get("direction", "LONG")
            item = display_item(t["coin"])
            if last is None:
                rows.append({"Item": item, "Interval": t["timeframe"],
                             "Scenario": display_direction(direction),
                             "Reference": f"{entry:,.6g}", "Boundary": f"{stop:,.6g}",
                             "Objective": f"{target:,.6g}", "Current": "—",
                             "Change": "—", "Status": "Unavailable"})
                continue
            if direction == "SHORT":
                boundary_hit, objective_hit = last >= stop, last <= target
                d_boundary, d_objective = (last / stop - 1) * 100, (last / target - 1) * 100
            else:
                boundary_hit, objective_hit = last <= stop, last >= target
                d_boundary, d_objective = (stop / last - 1) * 100, (target / last - 1) * 100
            flag = ("⛔ Boundary reached" if boundary_hit else
                    "✅ Objective reached" if objective_hit else "In progress")
            rows.append({
                "Item": item, "Interval": t["timeframe"],
                "Scenario": display_direction(direction),
                "Reference": f"{entry:,.6g}", "Boundary": f"{stop:,.6g}",
                "Objective": f"{target:,.6g}", "Current": f"{last:,.6g}",
                "Change": f"{_pnl_pct(entry, last, direction):+.2f}%",
                "To boundary": f"{d_boundary:+.2f}%",
                "To objective": f"{d_objective:+.2f}%",
                "Status": flag,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        for t in active:
            entry, stop = float(t["entry"]), float(t["stop"])
            distance_pct = abs(entry - stop) / entry * 100
            reference_size = (1000 * 0.01) / (distance_pct / 100)
            st.caption(
                f"**{display_item(t['coin'])}**: boundary is {distance_pct:.2f}% from the "
                f"reference; example 1% allocation of 1,000 units ≈ {reference_size:,.0f} units.")

        for t in active:
            c1, c2 = st.columns([1, 3])
            with c1:
                if st.button("Mark complete", key=f"close_{t['id']}", use_container_width=True):
                    last = _last_price(t["coin"], t["timeframe"])
                    if last is not None:
                        t.update(status="CLOSED", exit_price=last,
                                 closed_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                                 exit_reason="manually completed",
                                 pnl_pct=round(_pnl_pct(float(t["entry"]), last,
                                                        t.get("direction", "LONG")), 3))
                        _save_paper_trades(records)
                        st.rerun()
                    else:
                        st.error(f"Could not update {display_item(t['coin'])} right now.")
            with c2:
                st.caption(
                    f"{display_item(t['coin'])} · {t['timeframe']} · recorded {t.get('opened_at', '—')} · "
                    f"scenario {display_direction(t.get('direction', 'LONG'))}")

    if completed:
        st.markdown("#### 🏁 Completed")
        st.dataframe(pd.DataFrame([
            {
                "Item": display_item(t["coin"]),
                "Interval": t["timeframe"],
                "Scenario": display_direction(t.get("direction", "LONG")),
                "Reference": f"{float(t['entry']):,.6g}",
                "Final value": f"{float(t['exit_price']):,.6g}" if t.get("exit_price") is not None else "—",
                "Change": f"{t['pnl_pct']:+.2f}%" if t.get("pnl_pct") is not None else "—",
                "Recorded": t.get("opened_at", ""),
                "Completed": t.get("closed_at", ""),
                "Note": t.get("exit_reason", ""),
            } for t in completed
        ]), use_container_width=True, hide_index=True)

    st.caption(
        "Boundary and objective checks use the latest completed interval. "
        "Records are for personal monitoring and remain stored locally.")


def about_tab() -> None:
    st.markdown(
        """
## Workspace guide

This private workspace combines several independent data factors into a single
status label. Each factor can be positive, neutral, or negative. The final
status is based on the weighted balance of those factors and a consistency
check.

### Main sections

- **Overview** — current values, status labels, a neutral trend chart, and
  component assessments for each item.
- **Scenario review** — replay historical data with configurable thresholds.
- **Rolling validation** — select settings on one data section and evaluate
  them on a later unseen section.
- **Activity log** — store personal records locally and compare them with
  current values.

### Data handling

Only completed intervals are used for the primary status. The application
uses a secondary alignment check, an environment classification, and an
optional event filter. The event filter is disabled by default and can be
changed from the workspace controls.

### Reading the status

- **Priority Up** — strongly positive balance
- **Upward** — positive balance
- **Stable** — no clear direction
- **Downward** — negative balance
- **Priority Down** — strongly negative balance

The scenario and validation results are historical calculations, not
guarantees about future results.
        """
    )

def main() -> None:
    st.title("📊 Operations Workspace")
    st.caption(
        "Private data overview · trend analysis · scenario review · rolling validation")

    with st.sidebar:
        st.header("⚙️ Workspace controls")
        st.multiselect("Items", ITEM_OPTIONS, default=DEFAULT_ITEM_OPTIONS,
                       key="item_select")
        st.markdown("**Quick selection:**")
        b1, b2 = st.columns(2)
        if b1.button("All items", use_container_width=True):
            st.session_state["item_select"] = GROUP_OPTIONS["All items"]
            st.rerun()
        if b2.button("Primary group", use_container_width=True):
            st.session_state["item_select"] = GROUP_OPTIONS["Primary group"]
            st.rerun()
        b3, b4 = st.columns(2)
        if b3.button("Secondary group", use_container_width=True):
            st.session_state["item_select"] = GROUP_OPTIONS["Secondary group"]
            st.rerun()
        if b4.button("Recent group", use_container_width=True):
            st.session_state["item_select"] = GROUP_OPTIONS["Recent group"]
            st.rerun()
        timeframe = st.selectbox("Interval", TIMEFRAMES, index=2)
        limit = st.slider("Data points", 300, 1000, 500, step=100)
        use_patterns = st.checkbox(
            "Enable event filter", value=False,
            help="When enabled, a fresh reversal event can change a positive "
                 "status to Stable. The default is off because results vary by item.")
        if st.button("🔄 Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.caption("Data is cached for 5 minutes. Use Refresh to force a re-fetch.")

    tab_sig, tab_bt, tab_wf, tab_paper, tab_about = st.tabs(
        ["📊 Overview", "🧪 Scenario review", "🔬 Rolling validation",
         "📒 Activity log", "ℹ️ Guide"])

    with tab_sig:
        selected_labels = st.session_state.get("item_select", DEFAULT_ITEM_OPTIONS)
        items = [LABEL_TO_SYMBOL.get(label, label) for label in selected_labels]
        if not items:
            st.info("Select at least one item in the sidebar.")
        else:
            st.markdown(f"#### Overview — {len(items)} items, {timeframe}")
            st.dataframe(market_board(tuple(items), timeframe, limit, use_patterns),
                         use_container_width=True, hide_index=True)
            st.divider()
            for item in items:
                coin_section(item, timeframe, limit, use_patterns)
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
