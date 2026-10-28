"""Resample a company's daily frame to weekly/monthly using causal aggregation."""
import pandas as pd


AGG_MAP = {
    "Open": "first", "High": "max", "Low": "min", "Close": "last",
    "Volume": "sum",
    "MA7": "last", "MA21": "last", "MA50": "last",
    "RSI": "last", "MACD": "last", "MACD_Signal": "last", "MACD_Hist": "last",
    "BB_Upper": "last", "BB_Lower": "last", "BB_Middle": "last",
    "ATR": "last", "Volume_MA": "last", "Volume_Ratio": "last",
    "Momentum5": "last", "Momentum10": "last",
    # ARIMA / GARCH — only used when ARIMA is enabled
    "arima_fitted": "last", "arima_resid": "last", "garch_var": "last",
    # Targets — keep them so we can look up y values at the right time
    "Daily_Return": "last",
    "Weekly_Return": "last",
    "Monthly_Return": "last",
}


def _agg_for(df: pd.DataFrame) -> dict:
    """Return AGG_MAP restricted to columns that actually exist in df."""
    return {col: op for col, op in AGG_MAP.items() if col in df.columns}


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    agg = _agg_for(df)
    return df.resample("W").agg(agg).dropna()


def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    agg = _agg_for(df)
    return df.resample("ME").agg(agg).dropna()