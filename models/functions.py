# models/functions.py
from __future__ import annotations
import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, Set
import yfinance as yf
from src.adapters.yahoo_map import td_to_yahoo
from src.adapters.yahoo_map import td_to_yahoo, yahoo_alternates



# ---------- Data ----------
# models/functions.py  (only this function needs replacing)

def get_price_series(symbol: str, api_key: str, bars: int = 5000) -> pd.Series:
    """
    yfinance fetch with resilient fallbacks:
      1) map TD-style symbol -> Yahoo (td_to_yahoo)
      2) try period chain (short to long) to dodge Yahoo quirks (e.g., ^OMXS30)
      3) if still empty, try start-date download (10y back) via download()
      4) if still empty, try Ticker.history(start=...)
      5) if still empty, try alternates (yahoo_alternates)
    Returns a pd.Series (Close) or empty Series.
    """
    def _series_from_df(df: pd.DataFrame, name: str) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype="float64")
        # Prefer Adj Close if present; else Close
        if "Adj Close" in df:
            s = df["Adj Close"]
        elif "Close" in df:
            s = df["Close"]
        else:
            # maybe single-level Series already
            return pd.Series(dtype="float64")
        s = s.dropna()
        if bars:
            s = s.tail(bars)
        s.index = pd.to_datetime(s.index)
        s.name = name
        return s.astype(float)

    def _try_all_methods(yticker: str) -> pd.Series:
        if not yticker:
            return pd.Series(dtype="float64")

        # 2) Period chain (shorter first)
        period_chain = ["2y", "5y", "10y", "max"]
        if yticker.startswith("^"):            # indexes often dislike 'max'
            period_chain = ["10y", "5y", "2y", "max"]

        for per in period_chain:
            try:
                df = yf.download(
                    tickers=yticker,
                    period=per,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
            except Exception:
                df = None
            s = _series_from_df(df, symbol)
            if not s.empty:
                return s

        # 3) Start-date download (10y back)
        start = (dt.date.today() - dt.timedelta(days=365*10)).isoformat()
        try:
            df = yf.download(
                tickers=yticker,
                start=start,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception:
            df = None
        s = _series_from_df(df, symbol)
        if not s.empty:
            return s

        # 4) Ticker.history as last resort
        try:
            t = yf.Ticker(yticker)
            dfh = t.history(start=start, interval="1d", auto_adjust=True)
        except Exception:
            dfh = None
        s = _series_from_df(dfh, symbol)
        return s

    # 1) primary mapping
    primary = td_to_yahoo(symbol)
    s = _try_all_methods(primary)
    if not s.empty:
        return s

    # 5) alternates
    for alt in yahoo_alternates(symbol, primary):
        s_alt = _try_all_methods(alt)
        if not s_alt.empty:
            return s_alt

    return pd.Series(dtype="float64")




def compute_returns(prices: pd.DataFrame, kind: str = "Simple") -> pd.DataFrame:
    """Return daily returns (simple or log)."""
    if kind.lower() == "log":
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


# ---------- Weights ----------
def normalize_weights(weights: pd.Series) -> pd.Series:
    """Normalize any nonnegative weights to sum to 1. If all zero/NaN → equal weights."""
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    s = w.sum()
    if s <= 0:
        return pd.Series(np.ones(len(w)) / len(w), index=w.index)
    return w / s


# ---------- Calendar helpers ----------
def first_trading_days_each_month(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First available date per (year, month) from an index."""
    s = pd.Series(index=idx, data=1.0)
    return s.groupby([idx.year, idx.month]).head(1).index


def rebalance_months_for(freq: str) -> Set[int]:
    """Map rebalancing frequency to months set."""
    f = (freq or "").lower()
    if f == "none":
        return set()
    if f == "monthly":
        return set(range(1, 13))
    if f == "quarterly":
        return {3, 6, 9, 12}
    if f == "yearly":
        return {1}
    return set()


# ---------- Portfolio simulation ----------
def simulate_portfolio(
    prices_df: pd.DataFrame,
    target_w: pd.Series,
    initial: float,
    monthly: float,
    rebalancing: str,
) -> Dict[str, pd.Series | pd.DataFrame]:
    """
    Simulate a portfolio with initial investment, monthly contributions (on first trading day),
    and periodic rebalancing to target weights.
    Returns dict with keys: value, flows, shares, alloc, twr_returns
    """
    idx = prices_df.index
    cols = prices_df.columns

    # Dates for contributions & rebalancing
    month_firsts = first_trading_days_each_month(idx)
    contrib_dates = month_firsts[1:]  # skip month 1 (initial invested already)
    contrib_set = set(contrib_dates)

    rebal_months = rebalance_months_for(rebalancing)
    rebal_dates = pd.DatetimeIndex([d for d in month_firsts if d.month in rebal_months]) if rebal_months else pd.DatetimeIndex([])
    rebal_set = set(rebal_dates)

    # State
    shares = pd.DataFrame(index=idx, columns=cols, data=0.0)
    value = pd.Series(index=idx, dtype="float64")
    flows = pd.Series(0.0, index=idx, dtype="float64")  # + = external cash in

    # Day 0: buy according to target weights
    p0 = prices_df.iloc[0]
    init_shares = (initial * target_w) / p0
    shares.iloc[0] = init_shares
    value.iloc[0] = float((init_shares * p0).sum())
    flows.iloc[0] = initial

    # Iterate
    for i in range(1, len(idx)):
        d = idx[i]
        # carry forward
        shares.iloc[i] = shares.iloc[i-1].values

        # contribution?
        if d in contrib_set and monthly > 0:
            if rebalancing.lower() != "none" and d in rebal_set:
                # add, then rebalance
                port_val_before = float((shares.iloc[i] * prices_df.loc[d]).sum())
                total_val = port_val_before + monthly
                desired_value = total_val * target_w
                desired_shares = desired_value / prices_df.loc[d]
                shares.iloc[i] = desired_shares.values
                flows.loc[d] += monthly
            else:
                # buy pro-rata (no selling)
                add_shares = (monthly * target_w) / prices_df.loc[d]
                shares.iloc[i] = (shares.iloc[i] + add_shares).values
                flows.loc[d] += monthly

        # rebalancing (if not already done with the contribution above)
        if rebalancing.lower() != "none" and (d in rebal_set) and not (d in contrib_set and monthly > 0):
            port_val = float((shares.iloc[i] * prices_df.loc[d]).sum())
            desired_value = port_val * target_w
            desired_shares = desired_value / prices_df.loc[d]
            shares.iloc[i] = desired_shares.values

        # market value
        value.iloc[i] = float((shares.iloc[i] * prices_df.loc[d]).sum())

    # Allocation over time
    alloc = (shares * prices_df).div(value, axis=0).fillna(0.0)

    # Time-weighted daily returns (remove cash-flow effect)
    twr = pd.Series(index=idx, dtype="float64")
    twr.iloc[0] = 0.0
    for i in range(1, len(idx)):
        numer = value.iloc[i] - value.iloc[i-1] - flows.iloc[i]
        denom = value.iloc[i-1] if value.iloc[i-1] != 0 else np.nan
        twr.iloc[i] = numer / denom if pd.notnull(denom) else 0.0

    return {"value": value, "flows": flows, "shares": shares, "alloc": alloc, "twr_returns": twr}




