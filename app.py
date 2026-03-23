"""
╔══════════════════════════════════════════════════════════════╗
║  NIFTY SIGNALS PRO v3.0 — AI-Powered Trading Dashboard      ║
║  NVIDIA NIM LLM + Global Markets + BTST + Real-Time Alerts  ║
╚══════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd, numpy as np
from datetime import datetime
import pytz, logging, os

from config import *
from data_fetcher import *
from indicators import add_all_indicators, get_indicator_signals, calc_cpr, calc_orb_levels
from signal_engine import calculate_confluence_score, generate_trade_recommendation, select_best_strategy
from sentiment import calculate_news_sentiment_llm, calculate_news_sentiment, filter_relevant_headlines
from global_analysis import fetch_all_global_data, calculate_global_score, analyze_india_vix, analyze_indian_indices
from btst_predictor import predict_next_day_gap
from realtime_alerts import generate_realtime_alerts, get_exit_recommendation
from llm_engine import (
    get_nim_client, generate_trade_commentary, generate_btst_narrative,
    interpret_breaking_news, explain_alert, NVIDIANimClient,
    PRIMARY_MODEL, FAST_MODEL,
)

IST = pytz.timezone("Asia/Kolkata")
logging.basicConfig(level=logging.WARNING)

st.set_page_config(page_title="NiftySignals Pro v3.0 — AI Powered", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');
.stApp { font-family: 'DM Sans', sans-serif; }
.signal-card { border-radius: 12px; padding: 20px 24px; margin: 8px 0; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
.signal-buy { background: linear-gradient(135deg, #0d4f2b, #145a32); border-left: 5px solid #00e676; color: #e0f7e9; }
.signal-sell { background: linear-gradient(135deg, #5a1010, #6b1717); border-left: 5px solid #ff1744; color: #fce4ec; }
.signal-neutral { background: linear-gradient(135deg, #37474f, #455a64); border-left: 5px solid #ffc107; color: #fff8e1; }
.signal-action { font-size: 28px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.signal-detail { font-size: 14px; margin-top: 6px; opacity: 0.9; }
.alert-critical { background: #4a0000; border-left: 4px solid #ff1744; padding: 12px 16px; border-radius: 8px; margin: 6px 0; }
.alert-high { background: #3e2700; border-left: 4px solid #ff9800; padding: 12px 16px; border-radius: 8px; margin: 6px 0; }
.alert-medium { background: #1a237e; border-left: 4px solid #42a5f5; padding: 12px 16px; border-radius: 8px; margin: 6px 0; }
.btst-card { background: linear-gradient(135deg, #1a1a2e, #16213e); border: 1px solid #0f3460; border-radius: 12px; padding: 20px; }
.global-green { color: #00e676; font-weight: 600; }
.global-red { color: #ff1744; font-weight: 600; }
div[data-testid="stMetric"] { background: rgba(255,255,255,0.03); border-radius: 8px; padding: 10px 16px; border: 1px solid rgba(255,255,255,0.06); }
.llm-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 10px; font-weight: 700; }
.llm-on { background: #76ff03; color: #000; }
.llm-off { background: #616161; color: #fff; }
.ai-commentary { background: linear-gradient(135deg, #1a1a2e, #0f3460); border: 1px solid #533483; border-radius: 12px; padding: 20px; margin: 10px 0; font-style: italic; line-height: 1.7; }
.ai-tag { background: #7c4dff; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 700; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════ LLM INIT ════════════════════════
# Initialize NVIDIA NIM client from environment or sidebar input
if "nvidia_api_key" not in st.session_state:
    st.session_state.nvidia_api_key = os.environ.get("NVIDIA_API_KEY", "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")

# ═══════════════════════════ SIDEBAR ═════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Control Panel v3.0")

    # AI Status indicator
    nim_client = get_nim_client(st.session_state.nvidia_api_key)
    if nim_client.available:
        st.markdown('<span class="llm-badge llm-on">🤖 AI ONLINE</span> Llama 3.3 70B', unsafe_allow_html=True)
    else:
        st.markdown('<span class="llm-badge llm-off">⚪ AI OFFLINE</span> Using VADER fallback', unsafe_allow_html=True)

    st.markdown("---")
    symbol = st.selectbox("📊 Index", ["NIFTY50", "BANKNIFTY"])
    timeframe = st.selectbox("⏱️ Timeframe", list(TIMEFRAMES.keys()), index=1)
    capital = st.number_input("💰 Capital (₹)", 5000, 10000000, 10000, 5000)

    # ── Smart Risk Auto-Adjustment based on capital ──
    lot_cost_nifty = 75 * 50    # ~₹3,750 min for 1 lot
    lot_cost_bank = 30 * 100    # ~₹3,000 min for 1 lot
    min_lot_cost = lot_cost_bank if "BANK" in symbol else lot_cost_nifty

    if capital < 25000:
        auto_risk = 1.0
        st.caption(f"⚠️ Capital ₹{capital:,} is low for F&O. Risk auto-set to 1%. "
                   f"Min ~₹{min_lot_cost:,} needed for 1 lot.")
    elif capital < 50000:
        auto_risk = 1.5
    elif capital < 200000:
        auto_risk = 2.0
    else:
        auto_risk = 2.5

    risk_pct = st.slider("🎯 Risk % (auto-adjusted)", 0.5, 5.0, auto_risk, 0.5,
                          help=f"Auto-set to {auto_risk}% based on ₹{capital:,} capital")

    max_affordable_lots = max(1, int(capital * 0.8 / max(min_lot_cost, 1)))
    st.caption(f"Can afford ~{max_affordable_lots} lot(s) of {symbol}")

    st.markdown("---")
    current_position = st.selectbox("📌 Your Current Position", ["NONE", "BUY CE", "BUY PE"],
                                     help="Set AFTER you enter a trade on Groww")
    entry_prem = st.number_input("Entry Premium (₹)", 0, 5000, 0,
                                  help="Your actual buy price on Groww. Leave 0 if no trade.")
    if current_position != "NONE" and entry_prem == 0:
        st.warning("⚠️ Set your Entry Premium for accurate exit alerts!")
    st.markdown("---")
    auto_refresh = st.checkbox("🔄 Auto-refresh (60s)", False)
    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=60000, limit=500, key="ar")
        except: st.info("Install streamlit-autorefresh")
    fetch_options = st.checkbox("📋 Option Chain (NSE)", True)
    fetch_sent = st.checkbox("📰 News Sentiment", True)

    st.markdown("---")
    st.markdown("### 🤖 NVIDIA NIM AI")
    api_key_input = st.text_input(
        "API Key", value=st.session_state.nvidia_api_key,
        type="password", help="Get free key at build.nvidia.com",
    )
    if api_key_input != st.session_state.nvidia_api_key:
        st.session_state.nvidia_api_key = api_key_input
        st.rerun()

    use_llm = st.checkbox("🧠 Enable AI Analysis", value=bool(st.session_state.nvidia_api_key),
                           help="Uses Llama 3.3 70B for sentiment, commentary, alerts")
    if not st.session_state.nvidia_api_key:
        st.caption("Paste your NVIDIA NIM key to enable AI features. [Get free key →](https://build.nvidia.com)")

    # Refresh NIM client with current key
    nim_client = get_nim_client(st.session_state.nvidia_api_key) if use_llm else NVIDIANimClient("")

    st.markdown("---")
    st.caption("⚠️ Educational only. Trade at your own risk.")
    if st.button("🔄 Refresh Now", use_container_width=True, type="primary"):
        st.cache_data.clear(); st.rerun()

# ═══════════════════════ DATA LOADING ════════════════════════
@st.cache_data(ttl=60)
def load_price(sym, tf):
    c = TIMEFRAMES[tf]; return fetch_ohlcv(sym, c["interval"], c["period"])

@st.cache_data(ttl=120)
def load_oc(sym):
    raw = fetch_option_chain("NIFTY" if sym == "NIFTY50" else "BANKNIFTY")
    return parse_option_chain(raw) if raw else (pd.DataFrame(), {})

@st.cache_data(ttl=120)
def load_global(): return fetch_all_global_data()

@st.cache_data(ttl=300)
def load_vix_hist(): return fetch_vix_history("3mo")

@st.cache_data(ttl=120)
def load_indian_idx(): return analyze_indian_indices()

# ═══════════════════════ HEADER ══════════════════════════════
now = datetime.now(IST)
session = get_market_session()
market_open = is_market_open()

h1, h2, h3 = st.columns([3, 2, 2])
with h1:
    st.markdown("# 🤖 NiftySignals Pro v3.0")
    ai_tag = '<span class="llm-badge llm-on">AI ON</span>' if nim_client.available else '<span class="llm-badge llm-off">AI OFF</span>'
    st.markdown(f'{ai_tag} Global Markets • VIX • BTST • Real-Time Alerts • <b>Llama 3.3 70B</b>', unsafe_allow_html=True)
with h2:
    st.markdown(f"### {'🟢' if market_open else '🔴'} {session}")
    st.markdown(f"🕐 {now.strftime('%d %b %Y, %I:%M:%S %p IST')}")
with h3:
    vix_val = get_india_vix()
    vix_prev = get_vix_prev_close()
    vix_chg = round(((vix_val - vix_prev) / vix_prev) * 100, 2) if vix_prev > 0 else 0
    st.metric("India VIX", f"{vix_val:.2f}", f"{vix_chg:+.2f}%")

st.markdown("---")

# ═══════════════════════ LOAD ALL DATA ═══════════════════════
with st.spinner(f"Loading {symbol} data..."):
    df = load_price(symbol, timeframe)
if df.empty:
    st.error(f"❌ No data for {symbol}. Market may be closed."); st.stop()

df = add_all_indicators(df, timeframe)
current_price = float(df["Close"].iloc[-1])
prev_ohlc = get_previous_day_ohlc(symbol)
cpr = calc_cpr(prev_ohlc["high"], prev_ohlc["low"], prev_ohlc["close"]) if prev_ohlc["high"] > 0 else {}
orb = calc_orb_levels(df) if timeframe in ["Scalping", "Intraday"] else {}
indicator_signals = get_indicator_signals(df)

# Option chain
oc_df, oc_meta, pcr_data, max_pain, oi_sr, oi_bias = pd.DataFrame(), {}, {}, 0, {}, "NEUTRAL"
if fetch_options:
    with st.spinner("Fetching Option Chain..."):
        oc_df, oc_meta = load_oc(symbol)
    if not oc_df.empty:
        ne = oc_meta.get("expiry_dates", [""])[0]
        pcr_data = calculate_pcr(oc_df, ne)
        max_pain = calculate_max_pain(oc_df, ne)
        ul = oc_meta.get("underlying_value", current_price)
        oi_sr = get_oi_support_resistance(oc_df, ul, ne)
        oi_bias = analyze_oi_buildup(oc_df, ul, ne)

# Sentiment — LLM or VADER
news_score, news_label, headlines = 0.0, "N/A", []
if fetch_sent:
    with st.spinner("Analyzing sentiment..." + (" (AI)" if nim_client.available else " (VADER)")):
        news_score, news_label, headlines = calculate_news_sentiment_llm(nim_client if use_llm else None)

# ═══════════════════ GLOBAL MARKETS + VIX ════════════════════
with st.spinner("Fetching Global Markets..."):
    global_data = load_global()
global_score, global_label, global_details = calculate_global_score(global_data)

vix_hist = load_vix_hist()
vix_analysis = analyze_india_vix(vix_val, vix_hist)

# ═══════════════════ CONFLUENCE SCORE ════════════════════════
confluence_score, signal_label, component_scores = calculate_confluence_score(
    indicator_signals=indicator_signals, pcr_data=pcr_data, oi_bias=oi_bias,
    news_score=news_score, vix_level=vix_val or 15.0,
    global_score=global_score, vix_signal_score=vix_analysis.get("signal_score", 0),
)

trade = generate_trade_recommendation(
    symbol=symbol, current_price=current_price, confluence_score=confluence_score,
    signal_label=signal_label, df=df, oc_df=oc_df if not oc_df.empty else None,
    vix_level=vix_val or 15.0, capital=capital, timeframe=timeframe,
)

# ═══════════════════ REAL-TIME ALERTS ════════════════════════
alerts = generate_realtime_alerts(
    current_position=current_position, df=df,
    vix_current=vix_val, vix_prev=vix_prev,
    pcr_current=pcr_data.get("pcr_oi", 0), news_headlines=headlines,
    cpr_levels=cpr,
    oi_support=oi_sr.get("support", []), oi_resistance=oi_sr.get("resistance", []),
)
exit_rec = get_exit_recommendation(alerts, current_position, entry_prem,
                                   trade.get("entry_premium", 0))

# ═══════════════════════════════════════════════════════════════
#  🚨 REAL-TIME ALERTS (only during market hours)
# ═══════════════════════════════════════════════════════════════
if alerts and market_open:
    st.markdown("## 🚨 Real-Time Alerts")
    if current_position != "NONE":
        col_exit = st.columns([1])[0]
        with col_exit:
            ec = exit_rec.get("color", "#ffc107")
            st.markdown(f"""
            <div style="background: {ec}22; border: 2px solid {ec}; border-radius: 12px; padding: 16px; text-align: center;">
                <div style="font-size: 24px; font-weight: 700; color: {ec};">{exit_rec['action']}</div>
                <div style="font-size: 14px; margin-top: 4px;">{exit_rec['message']}</div>
            </div>""", unsafe_allow_html=True)

    for alert in alerts[:6]:
        sev = alert["severity"]
        cls = "alert-critical" if sev == "CRITICAL" else "alert-high" if sev == "HIGH" else "alert-medium"
        st.markdown(f"""
        <div class="{cls}">
            <b>{alert['emoji']} [{sev}] {alert['type']}</b> — {alert['message']}<br>
            <small>👉 <b>Action:</b> {alert['action']} | {alert['timestamp']}</small>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")
elif alerts and not market_open:
    # Show alerts as informational only when market is closed
    critical_alerts = [a for a in alerts if a["severity"] == "CRITICAL"]
    if critical_alerts:
        st.markdown("## 📋 Post-Market Observations")
        st.info("Market is closed. These alerts are for awareness only — no action needed right now.")
        for alert in critical_alerts[:3]:
            st.markdown(f"ℹ️ **{alert['type']}** — {alert['message']}")
        st.markdown("---")

# ═══════════════════════════════════════════════════════════════
#  📌 ACTIVE POSITION MONITOR (if position is set)
# ═══════════════════════════════════════════════════════════════
if current_position != "NONE" and entry_prem > 0:
    st.markdown("## 📌 Your Active Position")
    ap1, ap2, ap3 = st.columns(3)
    with ap1:
        st.metric("Position", current_position)
    with ap2:
        st.metric("Entry Premium", f"₹{entry_prem}")
    with ap3:
        if trade.get("entry_premium", 0) > 0:
            current_prem = trade["entry_premium"]
            pnl = current_prem - entry_prem
            pnl_pct = (pnl / entry_prem) * 100 if entry_prem > 0 else 0
            st.metric("Est. Current P&L", f"₹{pnl:+.1f}", f"{pnl_pct:+.1f}%")
        else:
            st.metric("Status", "🟢 Monitoring" if market_open else "💤 Market Closed")

# ═══════════════════════════════════════════════════════════════
#  🎯 MAIN SIGNAL
# ═══════════════════════════════════════════════════════════════
if current_position != "NONE":
    st.markdown("## 🎯 Latest Market Signal *(you have an active position)*")
    st.caption("This shows the current market direction. Use it to decide whether to HOLD or EXIT your active position, or plan your next trade.")
else:
    st.markdown("## 🎯 Live Trading Signal")
s1, s2 = st.columns([2, 1])
with s1:
    cc = "signal-buy" if "BUY" in signal_label else ("signal-sell" if "SELL" in signal_label else "signal-neutral")
    act = trade["action"] if trade["action"] != "NO TRADE" else "⏸️ NO TRADE — WAIT"
    st.markdown(f"""
    <div class="signal-card {cc}">
        <div class="signal-action">{act}</div>
        <div class="signal-detail">Signal: {signal_label} | Confluence: {confluence_score:+.3f} | Confidence: {trade['confidence']:.0f}% | Global: {global_label} | VIX: {vix_analysis['zone']}</div>
    </div>""", unsafe_allow_html=True)
with s2:
    st.metric(f"{symbol}", f"₹{current_price:,.2f}")
    if cpr: st.caption(f"Pivot: {cpr.get('pivot',0):,.2f}")
    if max_pain: st.caption(f"Max Pain: {max_pain:,.0f}")

# ═══════════════════════════════════════════════════════════════
#  💹 TRADE DETAILS
# ═══════════════════════════════════════════════════════════════
if trade["action"] != "NO TRADE":
    st.markdown("## 💹 Trade Setup — Execute on Groww")
    if trade.get("capital_warning"):
        st.warning(trade["capital_warning"])
    t1, t2, t3 = st.columns(3)
    lot_l = "NIFTY" if "BANK" not in symbol else "BANKNIFTY"
    with t1:
        st.markdown("### 📋 Order Details")
        st.markdown(f"""
| | |
|---|---|
| **Action** | **{trade['action']}** |
| **Strike** | **{trade['strike']}** {'CE' if 'CE' in trade['action'] else 'PE'} |
| **Entry** | ₹{trade['entry_premium']:.1f} |
| **Lots** | {trade['max_lots']} × {trade['lot_size']} units |
| **Investment** | ₹{trade['total_investment']:,.0f} |""")
    with t2:
        st.markdown("### 🎯 Targets & Stop-Loss")
        st.markdown(f"""
| | |
|---|---|
| 🔴 **SL** | **₹{trade['sl_premium']:.1f}** / {trade['sl_underlying']:,.2f} |
| 🟢 **Target 1** | **₹{trade['target1_premium']:.1f}** / {trade['target1_underlying']:,.2f} |
| 🟢 **Target 2** | **₹{trade['target2_premium']:.1f}** / {trade['target2_underlying']:,.2f} |""")
    with t3:
        st.markdown("### 📊 Risk-Reward")
        st.markdown(f"""
| | |
|---|---|
| **Risk** | ₹{trade['total_risk']:,.0f} ({trade['total_risk']/capital*100:.1f}%) |
| **R:R** | 1:{trade['risk_reward']:.1f} |
| **Profit T1** | ₹{trade['potential_profit_t1']:,.0f} |
| **Profit T2** | ₹{trade['potential_profit_t2']:,.0f} |""")

    with st.expander("📱 Groww Execution Steps", expanded=True):
        ot = "CE" if "CE" in trade["action"] else "PE"
        st.markdown(f"""
**Groww App:** F&O → {lot_l} → Options → Nearest weekly expiry → **{trade['strike']} {ot}** → BUY → ₹{trade['entry_premium']:.1f} → {trade['max_lots']} lot(s)

**After entry:** SL at ₹{trade['sl_premium']:.1f} | Book 50% at ₹{trade['target1_premium']:.1f} | Trail rest to ₹{trade['target2_premium']:.1f} | **Exit by 3:15 PM (intraday)**
""")
    for r in trade["reasoning"]: st.caption(f"• {r}")

    # ── AI TRADE COMMENTARY ──────────────────────────────────
    if nim_client.available:
        with st.spinner("🤖 AI generating trade commentary..."):
            commentary = generate_trade_commentary(
                nim_client, symbol, current_price, signal_label, confluence_score,
                trade["action"], trade["strike"], indicator_signals,
                global_label, vix_val or 15, pcr_data.get("pcr_oi", 0),
            )
        if commentary:
            st.markdown(f"""<div class="ai-commentary">
                <span class="ai-tag">🤖 AI ANALYSIS</span> <b>Llama 3.3 70B</b><br><br>
                {commentary}
            </div>""", unsafe_allow_html=True)
else:
    st.info("⏸️ No clear signal. Wait for confluence to build.")

# ═══════════════════════════════════════════════════════════════
#  🌍 GLOBAL MARKETS HEATMAP
# ═══════════════════════════════════════════════════════════════
st.markdown("## 🌍 Global Markets & Impact on Nifty")

gm1, gm2 = st.columns([1, 2])
with gm1:
    gc = "#00e676" if global_score > 0 else "#ff1744" if global_score < 0 else "#ffc107"
    st.markdown(f"""
    <div style="text-align:center; padding:20px; background:#1a1a2e; border-radius:12px; border:2px solid {gc};">
        <div style="font-size:12px; text-transform:uppercase; letter-spacing:2px; opacity:0.6;">Global Score</div>
        <div style="font-size:42px; font-weight:700; font-family:'JetBrains Mono'; color:{gc};">{global_score:+.3f}</div>
        <div style="font-size:16px; color:{gc}; font-weight:600;">{global_label}</div>
    </div>""", unsafe_allow_html=True)

with gm2:
    for gname, gdata in global_details.items():
        with st.expander(f"{'🟢' if gdata['direction']=='BULLISH' else '🔴' if gdata['direction']=='BEARISH' else '🟡'} {gname} (Weight: {gdata['weight']:.0%}) → Score: {gdata['score']:+.3f}"):
            for tk in gdata["tickers"]:
                c = "global-green" if tk["change_pct"] > 0 else "global-red"
                corr_txt = "↑Nifty" if tk["nifty_impact"] > 0 else "↓Nifty"
                st.markdown(f"{tk['status']} **{tk['ticker']}**: {tk['price']:,.2f} (<span class='{c}'>{tk['change_pct']:+.2f}%</span>) → {corr_txt} ({tk['nifty_impact']:+.3f})", unsafe_allow_html=True)

# Indian Indices
indian_idx = load_indian_idx()
if indian_idx:
    st.markdown("### 🇮🇳 Indian Indices")
    ic = st.columns(len(indian_idx))
    for i, (name, data) in enumerate(indian_idx.items()):
        with ic[i]:
            st.metric(name, f"{data['price']:,.2f}", f"{data['change_pct']:+.2f}%")

# ═══════════════════════════════════════════════════════════════
#  📊 INDIA VIX DEEP ANALYSIS
# ═══════════════════════════════════════════════════════════════
st.markdown("## 📊 India VIX Deep Analysis")
v1, v2, v3 = st.columns([1, 1, 2])
with v1:
    zc = vix_analysis.get("zone_color", "#ffc107")
    st.markdown(f"""
    <div style="text-align:center; padding:20px; background:#1a1a2e; border-radius:12px; border:2px solid {zc};">
        <div style="font-size:36px; font-weight:700; font-family:'JetBrains Mono'; color:{zc};">{vix_val:.2f}</div>
        <div style="font-size:14px; color:{zc};">{vix_analysis['zone']} ZONE</div>
        <div style="font-size:12px; margin-top:4px;">Trend: {vix_analysis['trend']}</div>
    </div>""", unsafe_allow_html=True)
with v2:
    st.metric("VIX 1-Day Change", f"{vix_analysis.get('change_1d', 0):+.2f}%")
    st.metric("VIX 5-Day Change", f"{vix_analysis.get('change_5d', 0):+.2f}%")
    st.metric("Percentile", f"{vix_analysis.get('percentile', 50):.0f}th")
with v3:
    st.markdown(f"**Strategy:** {vix_analysis['action']}")
    st.info(vix_analysis.get("strategy_advice", ""))
    if vix_analysis.get("spike_alert"):
        st.error("🚨 VIX SPIKE DETECTED — Tighten all stop-losses!")
    for d in vix_analysis.get("details", []):
        st.caption(d)

# VIX chart
if not vix_hist.empty:
    fig_vix = go.Figure()
    fig_vix.add_trace(go.Scatter(x=vix_hist.index, y=vix_hist["Close"], name="India VIX",
                                 line=dict(color="#e040fb", width=2), fill="tozeroy", fillcolor="rgba(224,64,251,0.1)"))
    fig_vix.add_hline(y=12, line_dash="dash", line_color="#00e676", annotation_text="Low (12)")
    fig_vix.add_hline(y=20, line_dash="dash", line_color="#ffc107", annotation_text="Normal High (20)")
    fig_vix.add_hline(y=25, line_dash="dash", line_color="#ff1744", annotation_text="High (25)")
    fig_vix.update_layout(height=250, template="plotly_dark", paper_bgcolor="#0e1117",
                          plot_bgcolor="#0e1117", margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
    st.plotly_chart(fig_vix, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  📊 PRICE CHART
# ═══════════════════════════════════════════════════════════════
st.markdown("## 📊 Price Chart & Indicators")
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.6, 0.2, 0.2],
                    subplot_titles=(f"{symbol} — {TIMEFRAMES[timeframe]['label']}", "RSI", "MACD"))

fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                              name="Price", increasing_line_color="#00e676", decreasing_line_color="#ff1744"), row=1, col=1)

if "EMA_9" in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_9"], name="EMA 9", line=dict(width=1, color="#42a5f5")), row=1, col=1)
if "EMA_21" in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_21"], name="EMA 21", line=dict(width=1, color="#ffa726")), row=1, col=1)

st_col, std_col = "ST_10_3.0", "STd_10_3.0"
if st_col in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df[st_col].where(df[std_col]==1), name="ST Bull",
                             line=dict(width=2, color="#00e676")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df[st_col].where(df[std_col]==-1), name="ST Bear",
                             line=dict(width=2, color="#ff1744")), row=1, col=1)

if "VWAP" in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"], name="VWAP",
                             line=dict(width=2, color="#e040fb", dash="dash")), row=1, col=1)

if cpr:
    for n, v, c in [("Pivot", cpr["pivot"], "white"), ("R1", cpr["r1"], "#66bb6a"), ("S1", cpr["s1"], "#ef5350")]:
        fig.add_hline(y=v, line_dash="dot", line_color=c, annotation_text=f"{n}:{v:.0f}", row=1, col=1)

if "RSI_7" in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI_7"], name="RSI 7", line=dict(width=1.5, color="#42a5f5")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

if "MACD" in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(width=1.5, color="#42a5f5")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(width=1.5, color="#ffa726")), row=3, col=1)
    colors = ["#00e676" if v >= 0 else "#ff1744" for v in df["MACD_Hist"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Hist", marker_color=colors), row=3, col=1)

fig.update_layout(height=750, template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                  showlegend=True, legend=dict(orientation="h", y=1.02), xaxis_rangeslider_visible=False,
                  margin=dict(l=10,r=10,t=40,b=10))
st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  📋 INDICATOR DASHBOARD
# ═══════════════════════════════════════════════════════════════
st.markdown("## 📋 All Indicator Signals")
cols = st.columns(4)
for i, (name, data) in enumerate(indicator_signals.items()):
    sig = data["signal"]
    e = "🟢" if sig > 0.2 else ("🔴" if sig < -0.2 else "🟡")
    with cols[i % 4]:
        st.markdown(f"{e} **{name.replace('_',' ').title()}** — {data['label']}")
        st.caption(data['detail'])

# Confluence chart
st.markdown("## 🧮 Confluence Score Breakdown")
fig_c = go.Figure()
comps = list(component_scores.items())
names = [c[0].replace("_"," ").title() for c in comps]
vals = [c[1] for c in comps]
wts = [STRATEGY_WEIGHTS.get(c[0], 0) for c in comps]
weighted = [v * w for v, w in zip(vals, wts)]
colors = ["#00e676" if v > 0 else "#ff1744" if v < 0 else "#ffc107" for v in vals]
fig_c.add_trace(go.Bar(y=names, x=vals, orientation="h", marker_color=colors,
                       text=[f"{v:+.2f} (×{w:.0%}={wv:+.3f})" for v, w, wv in zip(vals, wts, weighted)],
                       textposition="outside"))
fig_c.update_layout(height=380, template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    xaxis=dict(range=[-1.2, 1.2], title="Signal (-1 to +1)"), yaxis=dict(autorange="reversed"),
                    margin=dict(l=10,r=10,t=10,b=10))
st.plotly_chart(fig_c, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  📋 OPTION CHAIN & OI
# ═══════════════════════════════════════════════════════════════
if not oc_df.empty:
    st.markdown("## 📋 Option Chain & OI Analysis")
    o1, o2, o3, o4 = st.columns(4)
    with o1: st.metric("PCR (OI)", f"{pcr_data.get('pcr_oi',0):.3f}")
    with o2: st.metric("Max Pain", f"{max_pain:,.0f}")
    with o3:
        be = "🟢" if oi_bias == "BULLISH" else ("🔴" if oi_bias == "BEARISH" else "🟡")
        st.metric("OI Bias", f"{be} {oi_bias}")
    with o4: st.metric("Underlying", f"₹{oc_meta.get('underlying_value',0):,.2f}")

    sr1, sr2 = st.columns(2)
    with sr1:
        st.markdown("**🟢 OI Support**")
        for l in oi_sr.get("support", []): st.markdown(f"  • {l:,}")
    with sr2:
        st.markdown("**🔴 OI Resistance**")
        for l in oi_sr.get("resistance", []): st.markdown(f"  • {l:,}")

    # OI Chart
    ne = oc_meta.get("expiry_dates", [""])[0]
    ed = oc_df[oc_df["expiry"] == ne].copy() if ne else oc_df.copy()
    step = 50 if "BANK" not in symbol else 100
    atm = get_atm_strike(current_price, step)
    ed = ed[(ed["strike"] >= atm - 10*step) & (ed["strike"] <= atm + 10*step)]
    if not ed.empty:
        fig_oi = make_subplots(rows=1, cols=2, subplot_titles=("Open Interest", "Change in OI"))
        fig_oi.add_trace(go.Bar(x=ed["strike"], y=ed["ce_oi"], name="CE OI", marker_color="#ff1744", opacity=0.7), row=1, col=1)
        fig_oi.add_trace(go.Bar(x=ed["strike"], y=ed["pe_oi"], name="PE OI", marker_color="#00e676", opacity=0.7), row=1, col=1)
        fig_oi.add_trace(go.Bar(x=ed["strike"], y=ed["ce_chg_oi"], name="CE ΔOI", marker_color="#ff1744", opacity=0.7), row=1, col=2)
        fig_oi.add_trace(go.Bar(x=ed["strike"], y=ed["pe_chg_oi"], name="PE ΔOI", marker_color="#00e676", opacity=0.7), row=1, col=2)
        fig_oi.add_vline(x=atm, line_dash="dash", line_color="white", row=1, col=1)
        fig_oi.add_vline(x=atm, line_dash="dash", line_color="white", row=1, col=2)
        fig_oi.update_layout(height=350, template="plotly_dark", paper_bgcolor="#0e1117",
                             plot_bgcolor="#0e1117", barmode="group", margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_oi, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  🔮 BTST GAP PREDICTOR
# ═══════════════════════════════════════════════════════════════
st.markdown("## 🔮 BTST — Tomorrow's Gap Prediction")
st.caption("Best accuracy when checked between 3:00 PM - 3:30 PM IST")

us_data = {k: v for k, v in global_data.items() if k in ["SP500_FUT", "DOW_FUT", "NASDAQ_FUT"]}
asian_data = {k: v for k, v in global_data.items() if k in ["NIKKEI", "HANGSENG", "SHANGHAI", "STRAITS"]}

btst = predict_next_day_gap(
    us_futures_data=us_data, asian_data=asian_data,
    vix_current=vix_val, vix_prev_close=vix_prev,
    df_today=df, pcr_eod=pcr_data.get("pcr_oi", 0),
    indicator_signals=indicator_signals,
)

b1, b2 = st.columns([1, 2])
with b1:
    bc = "#00e676" if btst["score"] > 0 else "#ff1744" if btst["score"] < 0 else "#ffc107"
    st.markdown(f"""
    <div class="btst-card" style="text-align:center; border-color:{bc};">
        <div style="font-size:42px;">{btst['emoji']}</div>
        <div style="font-size:28px; font-weight:700; color:{bc}; font-family:'JetBrains Mono';">{btst['prediction']}</div>
        <div style="font-size:14px; margin-top:8px;">Score: {btst['score']:+.3f} | Confidence: {btst['confidence']:.0f}%</div>
        <div style="font-size:12px; margin-top:4px; opacity:0.6;">{btst['bullish_count']} bullish vs {btst['bearish_count']} bearish factors</div>
    </div>""", unsafe_allow_html=True)

with b2:
    st.markdown("### Factor Breakdown")
    for fname, fdata in btst["factors"].items():
        fc = "🟢" if fdata["score"] > 0 else "🔴" if fdata["score"] < 0 else "⚪"
        wt = BTST_WEIGHTS.get(fname, 0)
        st.markdown(f"{fc} **{fname.replace('_',' ').title()}** (wt:{wt:.0%}) — {fdata['detail']} → Impact: {fdata['score']:+.3f}")

if btst.get("btst_trade"):
    bt = btst["btst_trade"]
    st.success(f"""
    **🎯 BTST Trade:** {bt['action']}
    • **Entry:** {bt['entry_time']} | **Exit:** {bt['exit_time']}
    • **Strike:** {bt['strike']} | **SL:** {bt['sl']} | **Target:** {bt['target']}
    • **Confidence:** {bt['confidence']:.0f}%
    • {bt['detail']}
    """)
else:
    st.info("No confident BTST trade for tonight. Gap prediction uncertain.")

# AI BTST Narrative
if nim_client.available and btst.get("factors"):
    with st.spinner("🤖 AI explaining gap prediction..."):
        btst_narr = generate_btst_narrative(
            nim_client, btst["prediction"], btst["score"],
            btst["confidence"], btst["factors"],
        )
    if btst_narr:
        st.markdown(f"""<div class="ai-commentary">
            <span class="ai-tag">🤖 AI INSIGHT</span> <b>Gap Analysis</b><br><br>
            {btst_narr}
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  📰 NEWS SENTIMENT
# ═══════════════════════════════════════════════════════════════
if headlines:
    st.markdown("## 📰 News Sentiment")
    ns1, ns2 = st.columns([1, 3])
    with ns1:
        nc = "#00e676" if news_score > 0.1 else "#ff1744" if news_score < -0.1 else "#ffc107"
        st.markdown(f"""<div style="text-align:center; padding:20px;">
            <div style="font-size:36px; font-weight:700; font-family:'JetBrains Mono'; color:{nc};">{news_score:+.3f}</div>
            <div style="color:{nc}; font-weight:600;">{news_label}</div>
        </div>""", unsafe_allow_html=True)
    with ns2:
        for h in filter_relevant_headlines(headlines)[:8]:
            ic = "🟢" if h["sentiment"] > 0.1 else ("🔴" if h["sentiment"] < -0.1 else "⚪")
            engine_badge = f'<span class="ai-tag">AI</span>' if h.get("engine") == "LLM" else ""
            impact_badge = ""
            if h.get("impact") == "high":
                impact_badge = " 🔥"
            reasoning = ""
            if h.get("reasoning"):
                reasoning = f'\n> *{h["reasoning"]}*'
            st.markdown(f"{ic} {engine_badge}{h['title'][:110]}{impact_badge} `{h['sentiment']:+.3f}` — *{h['source']}*{reasoning}", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  📐 KEY LEVELS
# ═══════════════════════════════════════════════════════════════
st.markdown("## 📐 Key Levels")
l1, l2, l3 = st.columns(3)
with l1:
    st.markdown("### CPR")
    if cpr:
        for n, v in [("R2",cpr.get("r2",0)),("R1",cpr.get("r1",0)),("TC",cpr.get("tc",0)),
                      ("Pivot",cpr.get("pivot",0)),("BC",cpr.get("bc",0)),("S1",cpr.get("s1",0)),("S2",cpr.get("s2",0))]:
            c = "🟢" if v > current_price else "🔴"
            st.markdown(f"{c} **{n}**: {v:,.2f}")
        if cpr.get("is_narrow_cpr"): st.warning("⚡ NARROW CPR — Trending day!")
with l2:
    st.markdown("### ORB")
    if orb and orb.get("orb_high", 0) > 0:
        st.markdown(f"🔼 **High**: {orb['orb_high']:,.2f}")
        st.markdown(f"📍 **Mid**: {orb['orb_mid']:,.2f}")
        st.markdown(f"🔽 **Low**: {orb['orb_low']:,.2f}")
        if current_price > orb["orb_high"]: st.success("Above ORB — Bullish breakout")
        elif current_price < orb["orb_low"]: st.error("Below ORB — Bearish breakdown")
    else: st.info("ORB calculates after 9:45 AM")
with l3:
    st.markdown("### ATR Levels")
    if "ATR_14" in df.columns:
        atr = float(df["ATR_14"].iloc[-1]) if not pd.isna(df["ATR_14"].iloc[-1]) else 0
        st.markdown(f"📏 **ATR(14)**: {atr:,.2f}")
        st.markdown(f"🔼 +1.5 ATR: {current_price + 1.5*atr:,.2f}")
        st.markdown(f"🔽 -1.5 ATR: {current_price - 1.5*atr:,.2f}")

# ═══════════════════════════════════════════════════════════════
#  🧠 HOW SIGNALS ARE CALCULATED (EXPLAINER)
# ═══════════════════════════════════════════════════════════════
with st.expander("🧠 HOW THE OVERALL SIGNAL IS CALCULATED — Full Explanation"):
    st.markdown(f"""
### Signal Calculation Pipeline

The system uses **Weighted Confluence Scoring** — 11 independent signal sources are combined:

**STEP 1: Each indicator scores -1 to +1**

| Component | Weight | Current Score | Weighted |
|-----------|--------|---------------|----------|
""" + "\n".join([
    f"| {c[0].replace('_',' ').title()} | {STRATEGY_WEIGHTS.get(c[0],0):.0%} | {c[1]:+.3f} | {c[1]*STRATEGY_WEIGHTS.get(c[0],0):+.4f} |"
    for c in component_scores.items()
]) + f"""

**STEP 2: Sum weighted scores** = {confluence_score:+.3f}

**STEP 3: VIX adjustment** — VIX at {vix_val:.1f} {'reduces' if vix_val > 20 else 'no effect on'} confidence

**STEP 4: Signal → {signal_label}**
- Above +0.65 = STRONG BUY
- +0.45 to +0.65 = BUY
- -0.30 to +0.45 = NEUTRAL (no trade)
- -0.65 to -0.30 = SELL
- Below -0.65 = STRONG SELL

**STEP 5: Trade generated** → {trade['action']} at strike {trade['strike']}

**NEW in v2.0:**
- **Global Markets (14%)**: US futures, Asian markets, Crude, DXY, etc. currently scoring {global_score:+.3f} ({global_label})
- **VIX Analysis (12%)**: India VIX at {vix_val:.2f} ({vix_analysis['zone']} zone), scoring {vix_analysis.get('signal_score',0):+.3f}

**BTST Prediction** uses 7 factors (US futures 25%, Asian close 15%, FII/DII 15%, VIX trend 10%, technicals 15%, closing pattern 10%, PCR 10%) to predict gap-up/gap-down.

**Real-Time Alerts** monitor Supertrend flips, VWAP breaks, VIX spikes, OI shifts, and breaking news to tell you when to EXIT.
""")

# ═══════════════════════════════════════════════════════════════
#  🤖 AI ENGINE STATUS
# ═══════════════════════════════════════════════════════════════
if nim_client.available:
    st.markdown("## 🤖 AI Engine Status")
    stats = nim_client.stats
    a1, a2, a3, a4 = st.columns(4)
    with a1: st.metric("Status", "🟢 Online")
    with a2: st.metric("Model", "Llama 3.3 70B")
    with a3: st.metric("API Calls", stats["calls"])
    with a4: st.metric("Errors", stats["errors"])

# ═══════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; opacity:0.4; font-size:12px; padding:20px;">
    NiftySignals Pro v3.0 | AI-Powered FOSS Trading System | Python + Streamlit + NVIDIA NIM<br>
    🤖 AI: Llama 3.3 70B via NVIDIA NIM | Sentiment: LLM + VADER Fallback<br>
    ⚠️ NOT financial advice. Trade at your own risk. Always use stop-losses.<br>
    Data: Yahoo Finance, NSE India | © 2026 Open Source MIT
</div>""", unsafe_allow_html=True)
