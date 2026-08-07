"""
📈 Stock ML Predictor  —  Code_shashant
Free APIs: Alpha Vantage (get key at alphavantage.co) + yfinance fallback
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from api_helper import get_history, get_quote, yf_info
from ml_engine import train_and_evaluate, add_features, FEATURE_COLS

# ──────────────────────────────────────────────────────────
#  Page config
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock ML Predictor | Code_shashant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
#  CSS — dark navy / teal / orange  (Code_shashant palette)
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: #0A1628;
        font-family: 'Inter', sans-serif;
        color: #E8EDF5;
    }
    [data-testid="stSidebar"] {
        background: #0D1B2A !important;
        border-right: 1px solid #1E3A5F;
    }
    [data-testid="stSidebar"] * { color: #C5D5E8 !important; }

    .brand-header {
        background: linear-gradient(135deg, #0D1B2A 0%, #1E3A5F 100%);
        border-bottom: 2px solid #00C2CB;
        padding: 1.2rem 2rem;
        margin: -1rem -1rem 1.5rem -1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .brand-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.5px;
    }
    .brand-dot { color: #FF6B35; }
    .brand-sub { font-size: 0.8rem; color: #00C2CB; font-weight: 400; letter-spacing: 2px; text-transform: uppercase; }

    .metric-card {
        background: linear-gradient(145deg, #0D1B2A, #132035);
        border: 1px solid #1E3A5F;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #00C2CB; }
    .metric-label { font-size: 0.72rem; color: #6A8BA8; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 0.4rem; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #FFFFFF; font-family: 'JetBrains Mono', monospace; }
    .metric-sub   { font-size: 0.78rem; margin-top: 0.3rem; }
    .positive { color: #00E676; }
    .negative { color: #FF5252; }
    .neutral  { color: #00C2CB; }

    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #00C2CB;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #1E3A5F;
    }
    .company-card {
        background: #0D1B2A;
        border: 1px solid #1E3A5F;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    .company-card b { color: #00C2CB; }

    .prediction-box {
        background: linear-gradient(135deg, #0D2137 0%, #0A1E35 100%);
        border: 2px solid #00C2CB;
        border-radius: 14px;
        padding: 1.8rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .pred-label { font-size: 0.75rem; color: #6A8BA8; text-transform: uppercase; letter-spacing: 2px; }
    .pred-price { font-size: 3rem; font-weight: 700; color: #FFFFFF; font-family: 'JetBrains Mono', monospace; }
    .pred-change { font-size: 1.1rem; font-weight: 600; margin-top: 0.4rem; }

    .model-metric {
        background: #0D1B2A;
        border: 1px solid #1E3A5F;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
    }
    .model-metric .mlabel { font-size: 0.68rem; color: #6A8BA8; text-transform: uppercase; letter-spacing: 1px; }
    .model-metric .mvalue { font-size: 1.3rem; font-weight: 600; color: #E8EDF5; }

    .stButton > button {
        background: linear-gradient(135deg, #00C2CB, #0097A7);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.6rem 1.5rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    .stSelectbox label, .stTextInput label, .stSlider label { color: #8AAFC8 !important; font-size: 0.82rem !important; }
    .stSelectbox > div > div { background: #0D1B2A !important; border-color: #1E3A5F !important; color: #E8EDF5 !important; }

    .info-badge {
        display: inline-block;
        background: #132035;
        border: 1px solid #1E3A5F;
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.72rem;
        color: #8AAFC8;
        margin: 0.2rem;
    }
    .api-note {
        background: #0D2137;
        border-left: 3px solid #FF6B35;
        padding: 0.7rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.8rem;
        color: #AABFD0;
        margin: 0.8rem 0;
    }
    hr { border-color: #1E3A5F; }
    [data-testid="stMetric"] { background: #0D1B2A; border-radius: 8px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  Header
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
  <div>
    <div class="brand-title">📈 Stock ML Predictor <span class="brand-dot">●</span></div>
    <div class="brand-sub">Code_shashant ·  AI / ML Bootcamp Project</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  Sidebar — controls
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key = st.text_input(
        "Alpha Vantage API Key",
        value="",
        type="password",
        placeholder="Get free key at alphavantage.co",
        help="Free key — no credit card. Visit alphavantage.co/support/#api-key"
    )
    st.markdown("""<div class="api-note">
        🔑 <b>Free key</b> — sign up at <a href="https://alphavantage.co" target="_blank" style="color:#00C2CB">alphavantage.co</a><br>
        Without a key, yfinance data is used (also free).
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎯 Stock")

    POPULAR = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "RELIANCE.NS", "TCS.NS", "INFY.NS"]
    selected_popular = st.selectbox("Popular Stocks", ["Custom..."] + POPULAR)

    if selected_popular == "Custom...":
        symbol = st.text_input("Enter Symbol", value="AAPL", placeholder="e.g. AAPL, TSLA, RELIANCE.NS").upper().strip()
    else:
        symbol = selected_popular

    period_days = st.select_slider(
        "Historical Data Period",
        options=[90, 180, 365, 730, 1825],
        value=365,
        format_func=lambda x: {90:"3 Months",180:"6 Months",365:"1 Year",730:"2 Years",1825:"5 Years"}[x],
    )

    st.markdown("---")
    st.markdown("### 🤖 ML Model")

    model_name = st.selectbox(
        "Algorithm",
        ["Random Forest", "Gradient Boosting", "Linear Regression", "SVR"],
        index=0,
    )
    predict_days = st.slider("Predict N Days Ahead", min_value=1, max_value=30, value=5)

    run = st.button("🚀 Fetch & Predict", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#4A6A88; text-align:center; line-height:1.6">
        Models: Linear Regression · Random Forest<br>
        Gradient Boosting · SVR<br><br>
        <b style="color:#00C2CB">TechWithAshu</b> · ML Bootcamp
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
#  Plotly theme
# ──────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#0A1628",
    plot_bgcolor="#0A1628",
    font=dict(family="Inter", color="#8AAFC8"),
    xaxis=dict(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F"),
    yaxis=dict(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1E3A5F"),
    margin=dict(l=20, r=20, t=40, b=20),
)

# ──────────────────────────────────────────────────────────
#  Default landing view
# ──────────────────────────────────────────────────────────
if not run:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem;">
        <div style="font-size:4rem; margin-bottom:1rem;">📊</div>
        <h2 style="color:#FFFFFF; font-weight:700; font-size:1.8rem">ML-Powered Stock Analysis</h2>
        <p style="color:#6A8BA8; font-size:1rem; max-width:520px; margin:1rem auto">
            Select a stock symbol, choose an ML model, and click <b style="color:#00C2CB">Fetch &amp; Predict</b>
            to get AI-driven price forecasts with full technical analysis.
        </p>
        <div style="margin-top:2rem">
            <span class="info-badge">✅ Alpha Vantage API (Free)</span>
            <span class="info-badge">✅ yfinance Fallback</span>
            <span class="info-badge">✅ No Credit Card</span>
            <span class="info-badge">✅ 20+ Technical Features</span>
            <span class="info-badge">✅ 4 ML Algorithms</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────────────────
#  Data loading
# ──────────────────────────────────────────────────────────
with st.spinner(f"🔄 Fetching data for **{symbol}**..."):
    df = get_history(symbol, api_key, period_days)
    quote = get_quote(symbol, api_key)
    info  = yf_info(symbol)

if df.empty:
    st.error(f"❌ Could not fetch data for **{symbol}**. Please check the symbol and try again.")
    st.stop()

# ──────────────────────────────────────────────────────────
#  Company info + live quote row
# ──────────────────────────────────────────────────────────
company_name = info.get("name", symbol)
st.markdown(f"## {company_name}  <span style='font-size:1rem;color:#6A8BA8'>({symbol})</span>", unsafe_allow_html=True)
st.markdown(f'<span class="info-badge">📂 {info.get("sector","N/A")}</span> <span class="info-badge">🏭 {info.get("industry","N/A")}</span>', unsafe_allow_html=True)

if quote:
    price = quote.get("price", 0)
    chg   = quote.get("change", 0)
    chgp  = quote.get("change_pct", "0%")
    sign  = "positive" if chg >= 0 else "negative"
    arrow = "▲" if chg >= 0 else "▼"

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        ("PRICE", f"${price:,.2f}", f'<span class="{sign}">{arrow} {chgp}</span>'),
        ("OPEN",  f"${quote.get('open',0):,.2f}", ""),
        ("HIGH",  f"${quote.get('high',0):,.2f}", ""),
        ("LOW",   f"${quote.get('low',0):,.2f}", ""),
        ("VOLUME",f"{quote.get('volume',0):,}", ""),
    ]
    for col, (lbl, val, sub) in zip([c1,c2,c3,c4,c5], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{lbl}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

st.markdown("")

# ──────────────────────────────────────────────────────────
#  Tabs
# ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Chart", "📉 Technical Indicators", "🤖 ML Prediction", "🔍 Feature Analysis"])

# ══════════════════════════════════════════════
#  TAB 1 — Candlestick + Volume
# ══════════════════════════════════════════════
with tab1:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="OHLC",
        increasing_line_color="#00E676", decreasing_line_color="#FF5252",
        increasing_fillcolor="#00E676", decreasing_fillcolor="#FF5252",
    ), row=1, col=1)

    # MA overlays
    feat = add_features(df)
    fig.add_trace(go.Scatter(x=feat["date"], y=feat["ma_20"], name="MA 20",
        line=dict(color="#00C2CB", width=1.5), opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=feat["date"], y=feat["ma_50"], name="MA 50",
        line=dict(color="#FF6B35", width=1.5), opacity=0.8), row=1, col=1)

    # Volume bars
    colors_vol = ["#00E676" if r["close"] >= r["open"] else "#FF5252" for _, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="Volume",
        marker_color=colors_vol, opacity=0.6), row=2, col=1)

    fig.update_layout(
        title=f"{symbol} — Price History ({period_days}d)",
        height=520, xaxis_rangeslider_visible=False,
        **PLOT_LAYOUT
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Stats
    st.markdown('<div class="section-title">📊 Summary Statistics</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1: st.metric("Period High",  f"${df['high'].max():,.2f}")
    with s2: st.metric("Period Low",   f"${df['low'].min():,.2f}")
    with s3: st.metric("Avg Volume",   f"{df['volume'].mean():,.0f}")
    with s4: st.metric("Total Return", f"{((df['close'].iloc[-1]/df['close'].iloc[0])-1)*100:.2f}%")

# ══════════════════════════════════════════════
#  TAB 2 — Technical Indicators
# ══════════════════════════════════════════════
with tab2:
    feat = add_features(df)
    feat_clean = feat.dropna()

    # RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=feat_clean["date"], y=feat_clean["rsi"],
        name="RSI(14)", line=dict(color="#00C2CB", width=2)))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="#FF5252", opacity=0.6, annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00E676", opacity=0.6, annotation_text="Oversold (30)")
    fig_rsi.add_hrect(y0=30, y1=70, fillcolor="#00C2CB", opacity=0.05)
    fig_rsi.update_layout(title="RSI (14)", height=280, **PLOT_LAYOUT)
    st.plotly_chart(fig_rsi, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # MACD
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=feat_clean["date"], y=feat_clean["macd"],
            name="MACD", line=dict(color="#00C2CB", width=2)))
        fig_macd.add_trace(go.Scatter(x=feat_clean["date"], y=feat_clean["macd_signal"],
            name="Signal", line=dict(color="#FF6B35", width=1.5)))
        hist_colors = ["#00E676" if v >= 0 else "#FF5252" for v in feat_clean["macd_hist"]]
        fig_macd.add_trace(go.Bar(x=feat_clean["date"], y=feat_clean["macd_hist"],
            name="Histogram", marker_color=hist_colors, opacity=0.6))
        fig_macd.update_layout(title="MACD", height=280, **PLOT_LAYOUT)
        st.plotly_chart(fig_macd, use_container_width=True)

    with col_b:
        # Bollinger Bands
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=feat_clean["date"], y=feat_clean["bb_upper"],
            name="Upper BB", line=dict(color="#FF6B35", width=1, dash="dash")))
        fig_bb.add_trace(go.Scatter(x=feat_clean["date"], y=feat_clean["ma_20"],
            name="MA 20", line=dict(color="#00C2CB", width=2),
            fill="tonexty", fillcolor="rgba(0,194,203,0.05)"))
        fig_bb.add_trace(go.Scatter(x=feat_clean["date"], y=feat_clean["bb_lower"],
            name="Lower BB", line=dict(color="#FF6B35", width=1, dash="dash"),
            fill="tonexty", fillcolor="rgba(0,194,203,0.05)"))
        fig_bb.add_trace(go.Scatter(x=feat_clean["date"], y=feat_clean["close"],
            name="Close", line=dict(color="#FFFFFF", width=1.5)))
        fig_bb.update_layout(title="Bollinger Bands (20)", height=280, **PLOT_LAYOUT)
        st.plotly_chart(fig_bb, use_container_width=True)

    # Volatility
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=feat_clean["date"], y=feat_clean["volatility_20"]*100,
        name="20d Volatility (%)", line=dict(color="#00C2CB", width=2), fill="tozeroy",
        fillcolor="rgba(0,194,203,0.1)"))
    fig_vol.update_layout(title="Historical Volatility (20-day)", height=220, **PLOT_LAYOUT)
    st.plotly_chart(fig_vol, use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 3 — ML Prediction
# ══════════════════════════════════════════════
with tab3:
    with st.spinner(f"🧠 Training **{model_name}** model..."):
        results, err = train_and_evaluate(df, model_name, predict_days)

    if err:
        st.error(f"❌ {err}")
    else:
        curr = results["current_price"]
        pred = results["next_price"]
        chg_pred  = pred - curr
        chgp_pred = (chg_pred / curr) * 100
        direction = "▲ BUY signal" if chg_pred > 0 else "▼ SELL signal"
        dir_color = "#00E676" if chg_pred > 0 else "#FF5252"

        # Prediction box
        st.markdown(f"""
        <div class="prediction-box">
            <div class="pred-label">Predicted price in {predict_days} day(s) · {model_name}</div>
            <div class="pred-price">${pred:,.2f}</div>
            <div class="pred-change" style="color:{dir_color}">{direction}  ({chgp_pred:+.2f}%)</div>
            <div style="font-size:0.78rem; color:#4A6A88; margin-top:0.8rem">Current: ${curr:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Model metrics
        st.markdown('<div class="section-title">📐 Model Performance</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        metrics = [
            ("R² Score",  f"{results['r2']:.4f}", "Higher = better fit"),
            ("MAE",       f"${results['mae']:.2f}", "Mean Absolute Error"),
            ("RMSE",      f"${results['rmse']:.2f}", "Root Mean Sq. Error"),
            ("MAPE",      f"{results['mape']:.2f}%", "Mean Abs. % Error"),
        ]
        for col, (lbl, val, hint) in zip([m1,m2,m3,m4], metrics):
            with col:
                st.markdown(f"""
                <div class="model-metric">
                    <div class="mlabel">{lbl}</div>
                    <div class="mvalue">{val}</div>
                    <div style="font-size:0.65rem;color:#4A6A88;margin-top:0.3rem">{hint}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:0.78rem; color:#4A6A88; margin:0.8rem 0">
            Train samples: <b style="color:#8AAFC8">{results['train_size']}</b> &nbsp;|&nbsp;
            Test samples: <b style="color:#8AAFC8">{results['test_size']}</b>
        </div>""", unsafe_allow_html=True)

        # Actual vs Predicted chart
        st.markdown('<div class="section-title">📈 Actual vs Predicted</div>', unsafe_allow_html=True)
        dates_str = [str(d)[:10] for d in results["test_dates"]]
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=dates_str, y=results["y_test"],
            name="Actual", line=dict(color="#FFFFFF", width=2)))
        fig_pred.add_trace(go.Scatter(x=dates_str, y=results["y_pred"],
            name="Predicted", line=dict(color="#00C2CB", width=2, dash="dot")))
        fig_pred.update_layout(
            title=f"Actual vs Predicted Close Price — {model_name}",
            height=340, **PLOT_LAYOUT
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Residuals
        residuals = results["y_test"] - results["y_pred"]
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=dates_str, y=residuals,
            mode="markers+lines", name="Residuals",
            marker=dict(color=["#00E676" if r >= 0 else "#FF5252" for r in residuals], size=5),
            line=dict(color="#4A6A88", width=1)))
        fig_res.add_hline(y=0, line_color="#FF6B35", line_dash="dash")
        fig_res.update_layout(title="Residuals (Actual − Predicted)", height=240, **PLOT_LAYOUT)
        st.plotly_chart(fig_res, use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 4 — Feature Importance
# ══════════════════════════════════════════════
with tab4:
    # Always train to get feature importance
    with st.spinner("Computing feature importance..."):
        res_fi, _ = train_and_evaluate(df, "Random Forest", predict_days)

    if res_fi and res_fi.get("feat_importance"):
        fi = pd.DataFrame({
            "feature": list(res_fi["feat_importance"].keys()),
            "importance": list(res_fi["feat_importance"].values()),
        }).sort_values("importance", ascending=True)

        fig_fi = go.Figure(go.Bar(
            x=fi["importance"], y=fi["feature"],
            orientation="h",
            marker=dict(
                color=fi["importance"],
                colorscale=[[0, "#1E3A5F"], [0.5, "#00C2CB"], [1, "#FF6B35"]],
                showscale=False,
            ),
        ))
        fig_fi.update_layout(
            title="Feature Importance — Random Forest",
            height=520, **PLOT_LAYOUT,
            xaxis_title="Importance Score",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown('<div class="section-title">📘 Feature Descriptions</div>', unsafe_allow_html=True)
        feat_desc = {
            "rsi": "RSI(14) — momentum oscillator (0–100)", "ma_7": "7-day simple moving average",
            "ma_20": "20-day SMA (Bollinger midline)", "ma_50": "50-day SMA (trend filter)",
            "ema_12": "12-day EMA", "ema_26": "26-day EMA",
            "macd": "MACD line (EMA12 − EMA26)", "macd_signal": "MACD signal (9-day EMA of MACD)",
            "macd_hist": "MACD histogram", "bb_width": "Bollinger Band width (volatility proxy)",
            "bb_position": "Price position within Bollinger Bands",
            "returns_1d": "1-day return", "returns_5d": "5-day return", "returns_10d": "10-day return",
            "volatility_5": "5-day rolling std of returns", "volatility_20": "20-day rolling std of returns",
            "vol_ratio": "Volume vs 5-day avg volume", "hl_range": "High-Low range / close",
            "oc_ratio": "Open-Close / High-Low (candle body)", "dayofweek": "Day of the week (0=Mon)",
        }
        top5 = fi.tail(5)["feature"].tolist()[::-1]
        cols = st.columns(2)
        for i, (feat, desc) in enumerate(feat_desc.items()):
            is_top = "🔥 " if feat in top5 else ""
            with cols[i % 2]:
                st.markdown(f"**{is_top}{feat}** — {desc}")
    else:
        st.info("Feature importance is available for Random Forest and Gradient Boosting models.")
        # Correlation heatmap instead
        feat_df = add_features(df).dropna()[FEATURE_COLS + ["close"]]
        corr = feat_df.corr()["close"].drop("close").sort_values(ascending=False)
        fig_corr = go.Figure(go.Bar(
            x=corr.values, y=corr.index, orientation="h",
            marker_color=["#00E676" if v > 0 else "#FF5252" for v in corr.values],
        ))
        fig_corr.update_layout(title="Feature Correlation with Close Price", height=480, **PLOT_LAYOUT)
        st.plotly_chart(fig_corr, use_container_width=True)

# ──────────────────────────────────────────────────────────
#  Footer
# ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:0.75rem; color:#2A4A6A; padding:1rem 0">
    📈 Stock ML Predictor  ·  <b style="color:#00C2CB">Code_Shami</b>  ·  AI/ML Bootcamp Project<br>
    Data: Alpha Vantage API (free) + yfinance  ·  For educational purposes only. Not financial advice.
</div>
""", unsafe_allow_html=True)
