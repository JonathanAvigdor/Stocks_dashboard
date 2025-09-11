import pandas as pd
import numpy as np

# Large US stocks
US_STOCKS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # Nvidia
    "GOOGL", # Alphabet (Google)
    "AMZN",  # Amazon
    "META",  # Meta Platforms
    "TSLA",  # Tesla
    "JPM",   # JPMorgan Chase
    "JNJ",   # Johnson & Johnson
    "V",     # Visa
]

# Major indexes (global, Europe, Nordics, Israel)
INDEXES = [
    # US
    "SPY",    # S&P 500
    "QQQ",    # Nasdaq 100
    "DIA",    # Dow Jones Industrial Average
    "IWM",    # Russell 2000

    # Global
    "EFA",    # MSCI EAFE (Europe, Australasia, Far East)
    "EEM",    # MSCI Emerging Markets
    "ACWI",   # MSCI All Country World Index
    "VT",     # Vanguard Total World Stock

    # Europe
    "SX5E",   # EURO STOXX 50
    "DAX",    # German DAX
    "FTSE",   # FTSE 100 (UK)
    "CAC40",  # French CAC 40

    # Sweden
    "OMX",  # OMX Stockholm 30
    "OMXS100", # OMX Stockholm 100

    # Israel
    "TA35",    # TA-35
    "TA125",   # TA-125
]

# Combine them into a default list
DEFAULT_TICKERS = US_STOCKS + INDEXES

def to_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time")

def compute_moving_averages(df: pd.DataFrame, windows=(20, 50, 200)) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"SMA_{w}"] = out["close"].rolling(w, min_periods=1).mean()
    return out

def max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min() if not series.empty else np.nan
