from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="AI Swing Trader", page_icon="💹", layout="centered")

DEFAULT_WATCHLIST = [
    "SMH", "NVDA", "AVGO", "AMD", "TSM", "MU", "MRVL", "ARM",
    "MSFT", "META", "GOOGL", "PLTR", "QQQM", "VGT", "VUG"
]

# -----------------------------
# Indicators
# -----------------------------

def calculate_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(length).mean()


def pct_distance(price, level):
    if level == 0 or pd.isna(level):
        return np.nan
    return (price - level) / level * 100


@st.cache_data(ttl=1800)
def load_stock_data(ticker: str, period: str = "1y"):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    if len(df) < 80:
        return None

    df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["RSI14"] = calculate_rsi(df["Close"], 14)
    df["ATR14"] = calculate_atr(df, 14)
    df["ATR14_%"] = df["ATR14"] / df["Close"] * 100

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = macd - macd_signal

    df["Volume20Avg"] = df["Volume"].rolling(20).mean()
    df["High20"] = df["High"].rolling(20).max()
    df["Low20"] = df["Low"].rolling(20).min()
    ret = df["Close"].pct_change()
    df["Vol20_%"] = ret.rolling(20).std() * np.sqrt(252) * 100
    df["Vol60_%"] = ret.rolling(60).std() * np.sqrt(252) * 100
    return df.dropna()


@st.cache_data(ttl=1800)
def load_spy(period="1y"):
    return load_stock_data("SPY", period)


def links_for(ticker):
    symbol = quote(ticker.upper())
    return {
        "Yahoo": f"https://finance.yahoo.com/chart/{symbol}",
        "Fidelity": f"https://digital.fidelity.com/prgw/digital/research/quote/dashboard/summary?symbol={symbol}",
    }


# -----------------------------
# Scoring and trade plan
# -----------------------------

def compute_relative_strength(df: pd.DataFrame, spy: pd.DataFrame | None):
    if spy is None or spy.empty:
        return np.nan
    try:
        stock_20 = df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1
        spy_20 = spy["Close"].iloc[-1] / spy["Close"].iloc[-21] - 1
        return (stock_20 - spy_20) * 100
    except Exception:
        return np.nan


def trade_setup(ticker: str, capital: float, risk_pct: float, max_position_pct: float, target_mode: str):
    df = load_stock_data(ticker)
    if df is None:
        return None
    spy = load_spy()
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(latest["Close"])
    ema10 = float(latest["EMA10"])
    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])
    ema200 = float(latest["EMA200"])
    rsi = float(latest["RSI14"])
    atr = float(latest["ATR14"])
    atr_pct = float(latest["ATR14_%"])
    vol20 = float(latest["Vol20_%"])
    macd_hist = float(latest["MACD_Hist"])
    prev_macd = float(prev["MACD_Hist"])
    volume = float(latest["Volume"])
    volume_avg = float(latest["Volume20Avg"])
    volume_ratio = volume / volume_avg if volume_avg else 0
    high20_prev = float(df["High"].iloc[-21:-1].max())
    low20 = float(latest["Low20"])
    dist20 = pct_distance(price, ema20)
    dist50 = pct_distance(price, ema50)
    rs20 = compute_relative_strength(df, spy)

    trend_score = 0
    if price > ema20: trend_score += 25
    if ema20 > ema50: trend_score += 25
    if ema50 > ema200: trend_score += 20
    if macd_hist > prev_macd: trend_score += 15
    if not np.isnan(rs20) and rs20 > 0: trend_score += 15

    entry_score = 0
    # best swing entries: uptrend + pullback near EMA20, not too extended
    if -3 <= dist20 <= 2: entry_score += 35
    elif -5 <= dist20 < -3 or 2 < dist20 <= 4: entry_score += 20
    elif dist20 > 8: entry_score -= 20
    if 40 <= rsi <= 60: entry_score += 25
    elif 60 < rsi <= 68: entry_score += 10
    elif rsi > 72: entry_score -= 20
    if macd_hist > prev_macd: entry_score += 20
    if volume_ratio >= 1.1: entry_score += 10
    if price > high20_prev and volume_ratio >= 1.2: entry_score += 10  # breakout bonus

    risk_score = 0
    if atr_pct > 5: risk_score += 30
    elif atr_pct > 3.5: risk_score += 18
    elif atr_pct > 2.2: risk_score += 10
    if rsi > 70: risk_score += 25
    if dist20 > 8: risk_score += 25
    if price < ema50: risk_score += 20
    risk_score = min(100, risk_score)

    trade_score = round(max(0, min(100, trend_score * 0.45 + entry_score * 0.40 - risk_score * 0.20 + 20)), 1)

    if trade_score >= 75 and risk_score <= 55 and trend_score >= 60:
        action = "🟢 READY"
        action_note = "Good candidate for staged entry"
    elif trade_score >= 60 and trend_score >= 55:
        action = "🟡 NEAR ENTRY"
        action_note = "Watch for pullback or confirmation"
    elif risk_score >= 70 or dist20 > 8:
        action = "🔴 WAIT"
        action_note = "Too risky or extended"
    else:
        action = "⚪ WATCH"
        action_note = "No clear edge yet"

    # Entry levels: staged orders around EMA20/ATR zones
    aggressive_entry = round(min(price, price - 0.35 * atr), 2)
    normal_entry = round(max(ema20, price - 0.75 * atr), 2)
    strong_entry = round(max(ema50, price - 1.25 * atr), 2)
    planned_entry = normal_entry
    stop = round(min(ema50, planned_entry - 1.25 * atr), 2)
    if stop >= planned_entry:
        stop = round(planned_entry - 1.25 * atr, 2)

    if target_mode == "Percent targets":
        t1 = round(planned_entry * 1.05, 2)
        t2 = round(planned_entry * 1.10, 2)
        t3 = round(planned_entry * 1.15, 2)
    else:
        t1 = round(planned_entry + 1.0 * atr, 2)
        t2 = round(planned_entry + 1.5 * atr, 2)
        t3 = round(planned_entry + 2.0 * atr, 2)

    risk_dollars_total = capital * (risk_pct / 100)
    max_position_dollars = capital * (max_position_pct / 100)
    risk_per_share = max(planned_entry - stop, 0.01)
    shares_by_risk = int(risk_dollars_total // risk_per_share)
    shares_by_size = int(max_position_dollars // planned_entry)
    shares = max(0, min(shares_by_risk, shares_by_size))
    position_value = round(shares * planned_entry, 2)
    dollars_at_risk = round(shares * risk_per_share, 2)
    profit_t1 = round(shares * (t1 - planned_entry), 2)
    profit_t2 = round(shares * (t2 - planned_entry), 2)
    profit_t3 = round(shares * (t3 - planned_entry), 2)
    rr1 = round((t1 - planned_entry) / risk_per_share, 2)
    rr2 = round((t2 - planned_entry) / risk_per_share, 2)
    rr3 = round((t3 - planned_entry) / risk_per_share, 2)

    out = {
        "Ticker": ticker,
        "Action": action,
        "Action_Note": action_note,
        "Price": round(price, 2),
        "Trade_Score": trade_score,
        "Trend_Score": round(trend_score, 1),
        "Entry_Score": round(entry_score, 1),
        "Risk_Score": round(risk_score, 1),
        "RSI": round(rsi, 1),
        "ATR": round(atr, 2),
        "ATR_%": round(atr_pct, 2),
        "Vol20_%": round(vol20, 1),
        "EMA20": round(ema20, 2),
        "EMA50": round(ema50, 2),
        "EMA200": round(ema200, 2),
        "Dist_EMA20_%": round(dist20, 2),
        "Dist_EMA50_%": round(dist50, 2),
        "Volume_Ratio": round(volume_ratio, 2),
        "RelStrength20_%": round(rs20, 2) if not np.isnan(rs20) else None,
        "Aggressive_Entry": aggressive_entry,
        "Normal_Entry": normal_entry,
        "Strong_Entry": strong_entry,
        "Planned_Entry": planned_entry,
        "Stop": stop,
        "Target_1": t1,
        "Target_2": t2,
        "Target_3": t3,
        "Shares": shares,
        "Position_Value": position_value,
        "Dollars_At_Risk": dollars_at_risk,
        "Profit_T1": profit_t1,
        "Profit_T2": profit_t2,
        "Profit_T3": profit_t3,
        "RR1": rr1,
        "RR2": rr2,
        "RR3": rr3,
    }
    out.update(links_for(ticker))
    return out


def action_color(action: str):
    if "READY" in action:
        return "success"
    if "NEAR" in action:
        return "warning"
    if "WAIT" in action:
        return "error"
    return "info"


def show_candidate(row, key_prefix=""):
    with st.container(border=True):
        left, right = st.columns([1.3, 1])
        with left:
            st.markdown(f"### {row['Ticker']}  {row['Action']}")
            st.caption(row["Action_Note"])
        with right:
            st.metric("Trade Score", f"{row['Trade_Score']}/100")

        c1, c2, c3 = st.columns(3)
        c1.metric("Price", f"${row['Price']}")
        c2.metric("Timing", f"{row['Entry_Score']}/100")
        c3.metric("Risk", f"{row['Risk_Score']}/100")

        c4, c5, c6 = st.columns(3)
        c4.metric("RSI", row["RSI"])
        c5.metric("ATR%", f"{row['ATR_%']}%")
        c6.metric("EMA20 Dist", f"{row['Dist_EMA20_%']}%")

        st.markdown("**Trade Plan**")
        p1, p2 = st.columns(2)
        p1.metric("Planned Entry", f"${row['Planned_Entry']}")
        p2.metric("Stop", f"${row['Stop']}")

        t1, t2, t3 = st.columns(3)
        t1.metric("Target 1", f"${row['Target_1']}", f"${row['Profit_T1']}")
        t2.metric("Target 2", f"${row['Target_2']}", f"${row['Profit_T2']}")
        t3.metric("Target 3", f"${row['Target_3']}", f"${row['Profit_T3']}")

        s1, s2 = st.columns(2)
        s1.metric("Shares", int(row["Shares"]))
        s2.metric("Position", f"${row['Position_Value']}", f"Risk ${row['Dollars_At_Risk']}")

        with st.expander("Entry ladder + details", expanded=False):
            st.markdown(f"""
**Staged buy levels**
- Aggressive: **${row['Aggressive_Entry']}**
- Normal: **${row['Normal_Entry']}**
- Strong: **${row['Strong_Entry']}**

**Risk/Reward**
- T1 R/R: **{row['RR1']}**
- T2 R/R: **{row['RR2']}**
- T3 R/R: **{row['RR3']}**

**Trend data**
- Trend score: **{row['Trend_Score']}/100**
- Relative strength vs SPY, 20D: **{row['RelStrength20_%']}%**
- Volume ratio: **{row['Volume_Ratio']}x**
- Vol20: **{row['Vol20_%']}%**
""")
            l1, l2 = st.columns(2)
            l1.link_button("Yahoo Chart", row["Yahoo"], use_container_width=True)
            l2.link_button("Fidelity", row["Fidelity"], use_container_width=True)

        with st.expander("Chart", expanded=False):
            hist = load_stock_data(row["Ticker"], "6mo")
            if hist is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close"))
                fig.add_trace(go.Scatter(x=hist.index, y=hist["EMA20"], mode="lines", name="EMA20"))
                fig.add_trace(go.Scatter(x=hist.index, y=hist["EMA50"], mode="lines", name="EMA50"))
                fig.add_hline(y=row["Planned_Entry"], line_dash="dash", annotation_text="Entry")
                fig.add_hline(y=row["Stop"], line_dash="dash", annotation_text="Stop")
                fig.add_hline(y=row["Target_2"], line_dash="dash", annotation_text="Target 2")
                fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{key_prefix}_{row['Ticker']}")


# -----------------------------
# App UI
# -----------------------------

st.title("💹 AI Swing Trader")
st.caption("Short-term AI stock/ETF watchlist scanner and trade planner")
st.warning("Educational tool only. Not financial advice. Use limit orders and confirm prices with your broker.")

with st.expander("Trading settings", expanded=True):
    capital = st.number_input("Trading bucket", min_value=1000.0, max_value=1000000.0, value=10000.0, step=500.0, format="%.0f")
    risk_pct = st.slider("Max risk per trade", 0.25, 3.0, 1.0, 0.25)
    max_position_pct = st.slider("Max position size per ticker", 5, 60, 30, 5)
    target_mode = st.radio("Target method", ["ATR targets", "Percent targets"], horizontal=True)
    watchlist_text = st.text_area("Watchlist", value=", ".join(DEFAULT_WATCHLIST), height=95)

watchlist = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]
scan = st.button("🔍 Scan for Trades", use_container_width=True)

if scan or "trade_results" not in st.session_state:
    rows = []
    progress = st.progress(0)
    for i, ticker in enumerate(watchlist):
        row = trade_setup(ticker, capital, risk_pct, max_position_pct, target_mode)
        if row:
            rows.append(row)
        progress.progress((i + 1) / max(len(watchlist), 1))
    st.session_state["trade_results"] = pd.DataFrame(rows)
    st.session_state["last_scan"] = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %I:%M:%S %p %Z")

results = st.session_state.get("trade_results", pd.DataFrame())

if results.empty:
    st.info("No data loaded yet. Tap Scan for Trades.")
else:
    st.caption(f"Last scan: {st.session_state.get('last_scan', 'N/A')}")
    results = results.sort_values("Trade_Score", ascending=False)

    m1, m2 = st.columns(2)
    m1.metric("Ready", len(results[results["Action"].str.contains("READY")]))
    m2.metric("Near Entry", len(results[results["Action"].str.contains("NEAR")]))

    tabs = st.tabs(["Best", "Ready", "Near", "All", "Rules"])

    with tabs[0]:
        st.subheader("Top trade candidates")
        for i, row in results.head(5).iterrows():
            show_candidate(row, key_prefix=f"best_{i}")

    with tabs[1]:
        ready = results[results["Action"].str.contains("READY")]
        if ready.empty:
            st.info("No ready trades now. Preserve cash and wait.")
        for i, row in ready.iterrows():
            show_candidate(row, key_prefix=f"ready_{i}")

    with tabs[2]:
        near = results[results["Action"].str.contains("NEAR")]
        if near.empty:
            st.info("No near-entry trades now.")
        for i, row in near.iterrows():
            show_candidate(row, key_prefix=f"near_{i}")

    with tabs[3]:
        st.dataframe(results[[
            "Ticker", "Action", "Trade_Score", "Price", "Planned_Entry", "Stop",
            "Target_1", "Target_2", "Target_3", "Shares", "Position_Value",
            "Dollars_At_Risk", "RSI", "ATR_%", "Dist_EMA20_%", "RelStrength20_%"
        ]], use_container_width=True, hide_index=True)
        st.download_button(
            "Download trade plan CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="ai_swing_trade_plan.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tabs[4]:
        st.markdown("""
### Rules for the $10,000 trading bucket

**Goal:** capture repeatable 5%–15% swings, not perfect tops or bottoms.

**Buy only when most are true:**
- Trade Score is **75+**
- Trend Score is **60+**
- Risk Score is **below 55**
- Price is near EMA20 or pulling back in an uptrend
- RSI is roughly **40–60**

**Risk control:**
- Risk per trade should usually stay around **1%** of the trading bucket.
- Use the stop price before entering.
- If the stock gaps below stop, reassess quickly; do not average down blindly.

**Selling rule:**
- Sell part at Target 1.
- Sell more at Target 2.
- Let only a smaller piece try for Target 3.
""")
