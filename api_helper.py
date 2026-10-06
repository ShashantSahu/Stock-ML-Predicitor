"""
Stock data API helper
Primary  : Alpha Vantage
Fallback : yfinance
"""

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# Alpha Vantage
# ─────────────────────────────────────────────

AV_BASE = "https://www.alphavantage.co/query"


def av_quote(symbol: str, api_key: str) -> dict | None:
    """Latest quote from Alpha Vantage."""

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key,
    }

    try:
        r = requests.get(AV_BASE, params=params, timeout=15)
        r.raise_for_status()

        data = r.json()

        if "Note" in data:
            print("Alpha Vantage rate limit:", data["Note"])
            return None

        if "Information" in data:
            print("Alpha Vantage information:", data["Information"])
            return None

        if "Error Message" in data:
            print("Alpha Vantage error:", data["Error Message"])
            return None

        q = data.get("Global Quote", {})

        if not q:
            print("Alpha Vantage quote: Empty response")
            return None

        return {
            "symbol": q.get("01. symbol"),
            "price": float(q.get("05. price", 0)),
            "open": float(q.get("02. open", 0)),
            "high": float(q.get("03. high", 0)),
            "low": float(q.get("04. low", 0)),
            "volume": int(q.get("06. volume", 0)),
            "prev_close": float(q.get("08. previous close", 0)),
            "change": float(q.get("09. change", 0)),
            "change_pct": q.get("10. change percent", "0%"),
            "latest_day": q.get("07. latest trading day"),
        }

    except Exception as e:
        print("Alpha Vantage quote exception:", e)
        return None


def av_daily(
    symbol: str,
    api_key: str,
    outputsize: str = "compact"
) -> pd.DataFrame | None:
    """Daily OHLCV data from Alpha Vantage."""

    # IMPORTANT:
    # compact is used because it is suitable for the free API.
    # It returns the latest available daily records.

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": api_key,
    }

    try:
        r = requests.get(AV_BASE, params=params, timeout=20)
        r.raise_for_status()

        data = r.json()

        if "Note" in data:
            print("Alpha Vantage rate limit:", data["Note"])
            return None

        if "Information" in data:
            print("Alpha Vantage information:", data["Information"])
            return None

        if "Error Message" in data:
            print("Alpha Vantage error:", data["Error Message"])
            return None

        ts = data.get("Time Series (Daily)", {})

        if not ts:
            print("Alpha Vantage: No daily data returned.")
            print("Response keys:", list(data.keys()))
            return None

        rows = []

        for date_str, vals in ts.items():
            try:
                rows.append({
                    "date": pd.to_datetime(date_str),
                    "open": float(vals["1. open"]),
                    "high": float(vals["2. high"]),
                    "low": float(vals["3. low"]),
                    "close": float(vals["4. close"]),
                    "volume": int(vals["5. volume"]),
                })
            except (KeyError, ValueError, TypeError):
                continue

        if not rows:
            return None

        df = (
            pd.DataFrame(rows)
            .sort_values("date")
            .reset_index(drop=True)
        )

        print(
            f"Alpha Vantage SUCCESS: "
            f"{symbol} -> {len(df)} rows"
        )

        return df

    except Exception as e:
        print("Alpha Vantage daily exception:", e)
        return None


def av_rsi(
    symbol: str,
    api_key: str,
    interval: str = "daily"
) -> pd.DataFrame | None:
    """RSI from Alpha Vantage."""

    params = {
        "function": "RSI",
        "symbol": symbol,
        "interval": interval,
        "time_period": 14,
        "series_type": "close",
        "apikey": api_key,
    }

    try:
        r = requests.get(AV_BASE, params=params, timeout=15)
        r.raise_for_status()

        data = r.json()

        if "Note" in data:
            print("Alpha Vantage RSI rate limit:", data["Note"])
            return None

        if "Information" in data:
            print("Alpha Vantage RSI information:", data["Information"])
            return None

        if "Error Message" in data:
            print("Alpha Vantage RSI error:", data["Error Message"])
            return None

        ts = data.get("Technical Analysis: RSI", {})

        if not ts:
            print("Alpha Vantage RSI: No data returned.")
            return None

        rows = [
            {
                "date": pd.to_datetime(k),
                "RSI": float(v["RSI"])
            }
            for k, v in ts.items()
        ]

        return (
            pd.DataFrame(rows)
            .sort_values("date")
            .reset_index(drop=True)
        )

    except Exception as e:
        print("Alpha Vantage RSI exception:", e)
        return None


# ─────────────────────────────────────────────
# yFinance fallback
# ─────────────────────────────────────────────

def yf_history(
    symbol: str,
    period: str = "1y"
) -> pd.DataFrame | None:
    """Historical OHLCV via yfinance."""

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)

        if df.empty:
            print(f"yFinance: No data for {symbol}")
            return None

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        df = df.rename(
            columns={
                "stock splits": "splits",
                "capital gains": "capgains"
            }
        )

        df["date"] = pd.to_datetime(df["date"])

        if hasattr(df["date"].dt, "tz") and df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)

        return df[
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        ]

    except Exception as e:
        print("yFinance history error:", e)
        return None


def yf_info(symbol: str) -> dict:
    """Company information via yfinance."""

    try:
        t = yf.Ticker(symbol)
        info = t.info

        return {
            "name": info.get("longName", symbol),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low": info.get("fiftyTwoWeekLow", None),
            "description": info.get(
                "longBusinessSummary",
                ""
            ),
        }

    except Exception as e:
        print("yFinance info error:", e)
        return {}


# ─────────────────────────────────────────────
# Unified history fetch
# ─────────────────────────────────────────────

def get_history(
    symbol: str,
    api_key: str = "",
    period_days: int = 365
) -> pd.DataFrame:
    """
    Fetch stock history.

    Priority:
    1. Alpha Vantage
    2. yFinance fallback
    """

    df = None

    # ───── Alpha Vantage first ─────

    if (
        api_key
        and api_key.strip()
        and api_key.strip().lower() != "demo"
    ):

        print(f"Trying Alpha Vantage for {symbol}...")

        # IMPORTANT:
        # Always use compact for the free API.
        df = av_daily(
            symbol,
            api_key,
            "compact"
        )

        if df is not None and not df.empty:

            # If user requested a shorter period,
            # trim the available Alpha Vantage data.
            if period_days < 365:

                cutoff = (
                    datetime.now()
                    - timedelta(days=period_days)
                )

                df = df[
                    df["date"] >= cutoff
                ].reset_index(drop=True)

            print(
                f"Using Alpha Vantage data for {symbol}: "
                f"{len(df)} rows"
            )

            return df

    else:
        print("Alpha Vantage API key is EMPTY.")

    # ───── yFinance fallback ─────

    print(
        f"Alpha Vantage failed. "
        f"Trying yFinance for {symbol}..."
    )

    period_map = {
        30: "1mo",
        90: "3mo",
        180: "6mo",
        365: "1y",
        730: "2y",
        1825: "5y",
    }

    yf_period = min(
        period_map,
        key=lambda x: abs(x - period_days)
    )

    df = yf_history(
        symbol,
        period_map[yf_period]
    )

    if df is None:
        print(
            f"Both Alpha Vantage and "
            f"yFinance failed for {symbol}"
        )
        return pd.DataFrame()

    return df


# ─────────────────────────────────────────────
# Unified quote fetch
# ─────────────────────────────────────────────

def get_quote(
    symbol: str,
    api_key: str = ""
) -> dict:

    # ───── Alpha Vantage first ─────

    if (
        api_key
        and api_key.strip()
        and api_key.strip().lower() != "demo"
    ):

        print(
            f"Trying Alpha Vantage quote for {symbol}..."
        )

        q = av_quote(
            symbol,
            api_key
        )

        if q:
            print(
                f"Using Alpha Vantage quote for {symbol}"
            )
            return q

    # ───── yFinance fallback ─────

    print(
        f"Trying yFinance quote for {symbol}..."
    )

    try:

        t = yf.Ticker(symbol)
        h = t.history(period="2d")

        if not h.empty:

            latest = h.iloc[-1]

            prev = (
                h.iloc[-2]
                if len(h) > 1
                else latest
            )

            price = float(latest["Close"])
            prev_c = float(prev["Close"])
            chg = price - prev_c

            change_pct = (
                (chg / prev_c) * 100
                if prev_c != 0
                else 0
            )

            return {
                "symbol": symbol,
                "price": price,
                "open": float(latest["Open"]),
                "high": float(latest["High"]),
                "low": float(latest["Low"]),
                "volume": int(latest["Volume"]),
                "prev_close": prev_c,
                "change": chg,
                "change_pct": f"{change_pct:+.2f}%",
                "latest_day": str(h.index[-1].date()),
            }

    except Exception as e:
        print("yFinance quote error:", e)

    return {}
