"""
Stock data API helper
Primary  : Alpha Vantage (free key, no credit card — get yours at alphavantage.co)
Fallback : yfinance  (completely free, no key needed)
"""

import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
#  Alpha Vantage endpoints
# ─────────────────────────────────────────────
AV_BASE = "https://www.alphavantage.co/query"


def av_quote(symbol: str, api_key: str) -> dict | None:
    """Real-time quote from Alpha Vantage."""
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key,
    }
    try:
        r = requests.get(AV_BASE, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        q = data.get("Global Quote", {})
        if not q:
            return None
        return {
            "symbol":        q.get("01. symbol"),
            "price":         float(q.get("05. price", 0)),
            "open":          float(q.get("02. open", 0)),
            "high":          float(q.get("03. high", 0)),
            "low":           float(q.get("04. low", 0)),
            "volume":        int(q.get("06. volume", 0)),
            "prev_close":    float(q.get("08. previous close", 0)),
            "change":        float(q.get("09. change", 0)),
            "change_pct":    q.get("10. change percent", "0%"),
            "latest_day":    q.get("07. latest trading day"),
        }
    except Exception:
        return None


def av_daily(symbol: str, api_key: str, outputsize: str = "compact") -> pd.DataFrame | None:
    """Daily OHLCV from Alpha Vantage (compact = last 100 days)."""
    params = {
        "function":   "TIME_SERIES_DAILY",
        "symbol":     symbol,
        "outputsize": outputsize,
        "apikey":     api_key,
    }
    try:
        r = requests.get(AV_BASE, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        ts = data.get("Time Series (Daily)", {})
        if not ts:
            return None
        rows = []
        for date_str, vals in ts.items():
            rows.append({
                "date":   pd.to_datetime(date_str),
                "open":   float(vals["1. open"]),
                "high":   float(vals["2. high"]),
                "low":    float(vals["3. low"]),
                "close":  float(vals["4. close"]),
                "volume": int(vals["5. volume"]),
            })
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        return df
    except Exception:
        return None


def av_rsi(symbol: str, api_key: str, interval: str = "daily") -> pd.DataFrame | None:
    """RSI from Alpha Vantage technical indicators."""
    params = {
        "function":    "RSI",
        "symbol":      symbol,
        "interval":    interval,
        "time_period": 14,
        "series_type": "close",
        "apikey":      api_key,
    }
    try:
        r = requests.get(AV_BASE, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        ts = data.get("Technical Analysis: RSI", {})
        if not ts:
            return None
        rows = [{"date": pd.to_datetime(k), "RSI": float(v["RSI"])} for k, v in ts.items()]
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    except Exception:
        return None


# ─────────────────────────────────────────────
#  yfinance fallback (always free)
# ─────────────────────────────────────────────
def yf_history(symbol: str, period: str = "1y") -> pd.DataFrame | None:
    """Historical OHLCV via yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"stock splits": "splits", "capital gains": "capgains"})
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df[["date", "open", "high", "low", "close", "volume"]]
    except Exception:
        return None


def yf_info(symbol: str) -> dict:
    """Company info via yfinance."""
    try:
        t = yf.Ticker(symbol)
        info = t.info
        return {
            "name":        info.get("longName", symbol),
            "sector":      info.get("sector", "N/A"),
            "industry":    info.get("industry", "N/A"),
            "market_cap":  info.get("marketCap", 0),
            "pe_ratio":    info.get("trailingPE", None),
            "52w_high":    info.get("fiftyTwoWeekHigh", None),
            "52w_low":     info.get("fiftyTwoWeekLow", None),
            "description": info.get("longBusinessSummary", ""),
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────
#  Unified fetch  (AV first, yfinance fallback)
# ─────────────────────────────────────────────
def get_history(symbol: str, api_key: str = "", period_days: int = 365) -> pd.DataFrame:
    """Returns a clean OHLCV DataFrame regardless of source."""
    df = None

    # Try Alpha Vantage if key provided
    if api_key and api_key.strip() and api_key.strip() != "demo":
        outputsize = "full" if period_days > 100 else "compact"
        df = av_daily(symbol, api_key, outputsize)
        if df is not None and period_days < 365 * 5:
            cutoff = datetime.now() - timedelta(days=period_days)
            df = df[df["date"] >= cutoff].reset_index(drop=True)

    # Fallback: yfinance
    if df is None or df.empty:
        period_map = {
            30: "1mo", 90: "3mo", 180: "6mo",
            365: "1y", 730: "2y", 1825: "5y",
        }
        yf_period = min(period_map, key=lambda x: abs(x - period_days))
        df = yf_history(symbol, period_map[yf_period])

    if df is None:
        return pd.DataFrame()
    return df


def get_quote(symbol: str, api_key: str = "") -> dict:
    """Latest price, falls back to yfinance last close."""
    if api_key and api_key.strip() and api_key.strip() != "demo":
        q = av_quote(symbol, api_key)
        if q:
            return q

    # yfinance fallback
    try:
        t = yf.Ticker(symbol)
        h = t.history(period="2d")
        if not h.empty:
            latest = h.iloc[-1]
            prev   = h.iloc[-2] if len(h) > 1 else latest
            price  = float(latest["Close"])
            prev_c = float(prev["Close"])
            chg    = price - prev_c
            return {
                "symbol":     symbol,
                "price":      price,
                "open":       float(latest["Open"]),
                "high":       float(latest["High"]),
                "low":        float(latest["Low"]),
                "volume":     int(latest["Volume"]),
                "prev_close": prev_c,
                "change":     chg,
                "change_pct": f"{(chg/prev_c)*100:+.2f}%",
                "latest_day": str(h.index[-1].date()),
            }
    except Exception:
        pass
    return {}
# streamlit run app.py