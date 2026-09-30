# 📈 Stock ML Predictor — Code_shashant

A complete ML-powered stock analysis and price prediction app built with **Streamlit**, **scikit-learn**, and **free stock APIs**.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔑 Free API Setup

### Alpha Vantage (Recommended)
1. Go to → https://alphavantage.co/support/#api-key
2. Enter your email — get an API key **instantly**
3. **No credit card required**
4. Free tier: 25 requests/day, 5 per minute
5. Paste the key in the sidebar when the app starts

### yfinance (Automatic Fallback)
- No key needed — works automatically if no AV key is provided
- Pulls data directly from Yahoo Finance
- Unlimited historical data

---

## 📦 Project Structure

```
stock_ml_app/
├── app.py          ← Main Streamlit UI
├── api_helper.py   ← Alpha Vantage + yfinance data layer
├── ml_engine.py    ← Feature engineering + ML models
├── requirements.txt
└── README.md
```

---

## 🤖 ML Models Available

| Model              | Best For                          |
|--------------------|-----------------------------------|
| Random Forest      | Robust, handles non-linearity     |
| Gradient Boosting  | High accuracy, slower training    |
| Linear Regression  | Baseline, fast, interpretable     |
| SVR                | Good with small datasets          |

---

## 📊 Features Engineered (20+)

- Moving Averages: MA7, MA20, MA50, EMA12, EMA26
- MACD: MACD line, Signal, Histogram
- RSI (14-day)
- Bollinger Bands: Width, Position
- Price Returns: 1d, 5d, 10d
- Volatility: 5d, 20d rolling std
- Volume Ratio
- Candlestick features: HL range, OC ratio
- Day of Week

---

