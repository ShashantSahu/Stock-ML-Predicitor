"""
ML Engine — feature engineering + model training + prediction
Models available: Linear Regression, Random Forest, SVR, Gradient Boosting
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─────────────────────────────────────────────
#  Feature Engineering
# ─────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as ML features."""
    df = df.copy()

    # Moving averages
    df["ma_7"]  = df["close"].rolling(7).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()

    # Exponential moving averages
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["macd"]        = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # RSI (14-day)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.rolling(14).mean()
    avg_l = loss.rolling(14).mean()
    rs    = avg_g / (avg_l + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_std       = df["close"].rolling(20).std()
    df["bb_upper"]    = df["ma_20"] + 2 * rolling_std
    df["bb_lower"]    = df["ma_20"] - 2 * rolling_std
    df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["ma_20"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # Price momentum
    df["returns_1d"]  = df["close"].pct_change(1)
    df["returns_5d"]  = df["close"].pct_change(5)
    df["returns_10d"] = df["close"].pct_change(10)

    # Volatility
    df["volatility_5"]  = df["returns_1d"].rolling(5).std()
    df["volatility_20"] = df["returns_1d"].rolling(20).std()

    # Volume features
    df["vol_ma_5"]    = df["volume"].rolling(5).mean()
    df["vol_ratio"]   = df["volume"] / (df["vol_ma_5"] + 1e-9)

    # Candle features
    df["hl_range"]    = (df["high"] - df["low"]) / df["close"]
    df["oc_ratio"]    = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)

    # Day-of-week
    df["dayofweek"]   = pd.to_datetime(df["date"]).dt.dayofweek

    return df


FEATURE_COLS = [
    "ma_7", "ma_20", "ma_50", "ema_12", "ema_26",
    "macd", "macd_signal", "macd_hist",
    "rsi", "bb_width", "bb_position",
    "returns_1d", "returns_5d", "returns_10d",
    "volatility_5", "volatility_20",
    "vol_ratio", "hl_range", "oc_ratio", "dayofweek",
]


# ─────────────────────────────────────────────
#  Model Training & Evaluation
# ─────────────────────────────────────────────
MODELS = {
    "Linear Regression":   LinearRegression(),
    "Random Forest":       RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR":                 SVR(kernel="rbf", C=100, epsilon=0.1),
}


def prepare_data(df: pd.DataFrame, predict_days: int = 5):
    """
    Prepare X, y for regression:
    y = close price `predict_days` into the future
    """
    df = add_features(df)
    df["target"] = df["close"].shift(-predict_days)
    df = df.dropna(subset=FEATURE_COLS + ["target"])

    X = df[FEATURE_COLS].values
    y = df["target"].values
    dates = df["date"].values

    return X, y, dates, df


def train_and_evaluate(df: pd.DataFrame, model_name: str, predict_days: int = 5):
    """Train selected model and return metrics + predictions."""
    X, y, dates, feat_df = prepare_data(df, predict_days)

    if len(X) < 60:
        return None, "Not enough data (need at least 60 rows after feature engineering)."

    X_train, X_test, y_train, y_test, d_train, d_test = train_test_split(
        X, y, dates, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = MODELS[model_name]
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-9))) * 100

    # Predict next price
    last_features = feat_df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
    last_scaled   = scaler.transform(last_features)
    next_price    = float(model.predict(last_scaled)[0])

    # Feature importance (RF / GB only)
    feat_importance = None
    if hasattr(model, "feature_importances_"):
        feat_importance = dict(zip(FEATURE_COLS, model.feature_importances_))

    results = {
        "model":          model_name,
        "predict_days":   predict_days,
        "mae":            round(mae, 4),
        "rmse":           round(rmse, 4),
        "r2":             round(r2, 4),
        "mape":           round(mape, 2),
        "next_price":     round(next_price, 4),
        "y_test":         y_test,
        "y_pred":         preds,
        "test_dates":     d_test,
        "feat_importance": feat_importance,
        "current_price":  float(feat_df["close"].iloc[-1]),
        "train_size":     len(X_train),
        "test_size":      len(X_test),
    }
    return results, None
