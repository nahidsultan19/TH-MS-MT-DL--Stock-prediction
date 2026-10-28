"""Add causal technical indicators and returns. All rolling ops are backward-looking."""
import numpy as np
import pandas as pd


PRICE_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "MA7", "MA21", "MA50", "RSI",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Lower", "BB_Middle",
    "ATR", "Volume_MA", "Volume_Ratio",
    "Momentum5", "Momentum10",
]


def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["Scrip", "Date"])
    g = df.groupby("Scrip")

    # Returns
    df["Daily_Return"]   = g["Close"].pct_change() * 100
    df["Weekly_Return"]  = g["Close"].pct_change(5) * 100
    df["Monthly_Return"] = g["Close"].pct_change(21) * 100

    # Moving averages
    df["MA7"]  = g["Close"].transform(lambda x: x.rolling(7,  min_periods=1).mean())
    df["MA21"] = g["Close"].transform(lambda x: x.rolling(21, min_periods=1).mean())
    df["MA50"] = g["Close"].transform(lambda x: x.rolling(50, min_periods=1).mean())

    # RSI
    df["RSI"] = g["Close"].transform(_rsi)

    # MACD
    ema12 = g["Close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = g["Close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = g["MACD"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger
    df["BB_Middle"] = g["Close"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    bb_std = g["Close"].transform(lambda x: x.rolling(20, min_periods=1).std().fillna(0))
    df["BB_Upper"] = df["BB_Middle"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Middle"] - 2 * bb_std

    # ATR
    prev_close = g["Close"].shift(1).fillna(df["Close"])
    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum((df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()),
    )
    df["ATR"] = tr.groupby(df["Scrip"]).transform(lambda x: x.rolling(14, min_periods=1).mean())

    # Volume
    df["Volume_MA"] = g["Volume"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_MA"] + 1e-10)

    # Momentum
    df["Momentum5"]  = g["Close"].transform(lambda x: x.pct_change(5)  * 100)
    df["Momentum10"] = g["Close"].transform(lambda x: x.pct_change(10) * 100)

    return df.dropna().reset_index(drop=True)