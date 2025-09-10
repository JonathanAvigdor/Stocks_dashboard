import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Optional, Tuple

from src.api.twelve_data import TwelveDataClient
from src.api.frankfurter import FrankfurterClient

# -----------------------
# Page config & title
# -----------------------
st.set_page_config(page_title="Stocks & FX Dashboard", layout="wide")
st.title("📊 Stocks & FX Dashboard")
st.caption("Fast snapshot of markets: KPIs • Watchlist • FX • Top movers")

# -----------------------
# API keys / clients
# -----------------------
td_key = st.secrets.get("TWELVEDATA_API_KEY", "")
if not td_key or td_key == "YOUR_TWELVE_DATA_KEY":
    st.error("Please set TWELVEDATA_API_KEY in .streamlit/secrets.toml")
    st.stop()

td = TwelveDataClient(td_key)
fx = FrankfurterClient()

# -----------------------
# Refresh control (sidebar)
# -----------------------
with st.sidebar:
    st.header("Controls")
    if "refresh_counter" not in st.session_state:
        st.session_state["refresh_counter"] = 0
    if st.button("🔄 Refresh data"):
        st.session_state["refresh_counter"] += 1

    # Default watchlist
    DEFAULT_TICKERS = ["SPY", "QQQ", "EFA", "EWD", "AAPL", "MSFT", "NVDA"]
    watchlist = st.multiselect("Watchlist", DEFAULT_TICKERS, default=DEFAULT_TICKERS)

    # FX quick board
    fx_pairs = st.multiselect(
        "FX pairs",
        ["USD/EUR", "EUR/USD", "USD/SEK", "SEK/EUR", "GBP/USD", "EUR/SEK"],
        default=["USD/SEK", "EUR/USD", "EUR/SEK"]
    )

# -----------------------
# Helpers (cached)
# -----------------------
@st.cache_data(ttl=180, show_spinner=False)
def td_last_two(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """Fetch last 2 daily closes for a symbol."""
    df = td.daily(symbol, interval="1day", outputsize=2)
    if df is None or df.empty or len(df) < 2:
        return None, None
    closes = df["close"].to_list()
    return float(closes[-2]), float(closes[-1])

@st.cache_data(ttl=180, show_spinner=False)
def td_recent_series(symbol: str, bars: int = 30) -> pd.Series:
    """Fetch recent daily closes for sparkline."""
    df = td.daily(symbol, interval="1day", outputsize=bars)
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    s = df.set_index("time")["close"]
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s

@st.cache_data(ttl=180, show_spinner=False)
def td_lookbacks_pct(symbol: str, lookbacks=(1, 5, 21, 63)) -> dict:
    """Return last price and % changes for given lookback days (trading)."""
    max_lb = max(lookbacks) + 2
    s = td_recent_series(symbol, bars=max(60, max_lb + 5))
    if s.empty or len(s) < max_lb:
        return {"last": np.nan, **{f"{lb}d": np.nan for lb in lookbacks}}
    out = {"last": float(s.iloc[-1])}
    for lb in lookbacks:
        if len(s) > lb and s.iloc[-lb-1] != 0:
            out[f"{lb}d"] = float(s.iloc[-1] / s.iloc[-lb-1] - 1.0)
        else:
            out[f"{lb}d"] = np.nan
    return out

def _fx_series_from_timeseries(hist, quote: str) -> pd.Series:
    """Build a pandas Series of rates from Frankfurter `timeseries` output in any supported shape."""
    try:
        if isinstance(hist, dict) and "rates" in hist:
            ser = pd.Series(
                {pd.to_datetime(k): float(v.get(quote)) for k, v in hist["rates"].items()}
            ).sort_index()
            return ser.astype(float)
        elif isinstance(hist, pd.DataFrame):
            if quote in hist.columns:
                ser = hist[quote].copy()
                if not isinstance(ser.index, pd.DatetimeIndex):
                    if "date" in hist.columns:
                        ser.index = pd.to_datetime(hist["date"])
                    else:
                        ser.index = pd.to_datetime(ser.index)
                return ser.sort_index().astype(float)
            elif "rates" in hist.columns:
                ser = hist["rates"].apply(lambda d: float(d.get(quote)))
                if "date" in hist.columns:
                    ser.index = pd.to_datetime(hist["date"])
                else:
                    ser.index = pd.to_datetime(ser.index)
                return ser.sort_index().astype(float)
    except Exception:
        pass
    return pd.Series(dtype="float64")

@st.cache_data(ttl=120, show_spinner=False)
def fx_latest_pair(pair: str) -> dict:
    """
    pair format 'BASE/QUOTE' -> latest rate & approx 1D % change.
    Works whether FrankfurterClient returns dict or DataFrame.
    """
    base, quote = pair.split("/")

    # Latest spot
    rate, date_str = np.nan, ""
    latest = fx.latest(base=base, symbols=quote)
    try:
        if isinstance(latest, dict):
            rate = float(latest.get("rates", {}).get(quote))
            date_str = latest.get("date", "")
        elif isinstance(latest, pd.DataFrame):
            if quote in latest.columns:
                rate = float(latest[quote].iloc[-1])
            elif "rates" in latest.columns:
                rate = float(latest["rates"].iloc[-1].get(quote))
            date_str = str(latest.index[-1]) if latest.index.size else ""
    except Exception:
        rate, date_str = np.nan, ""

    # Approx 1D change from a tiny history window
    chg_1d = np.nan
    try:
        end_dt = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(end_dt):
            end_dt = pd.Timestamp.today().normalize()
        start_dt = end_dt - pd.Timedelta(days=5)

        hist = fx.timeseries(start=start_dt.date(), end=end_dt.date(), base=base, symbols=quote)
        ser = _fx_series_from_timeseries(hist, quote)
        if len(ser) >= 2 and ser.iloc[-2] != 0:
            chg_1d = float(ser.iloc[-1] / ser.iloc[-2] - 1.0)
    except Exception:
        chg_1d = np.nan

    return {"pair": pair, "rate": rate, "chg_1d": chg_1d, "date": str(date_str)}

# -----------------------
# KPI ROW (4–6 metrics)
# -----------------------
kpi_cols = st.columns(5)

def kpi_for(symbol: str, col, label: Optional[str] = None):
    prev, last = td_last_two(symbol)
    if prev is None or last is None:
        col.metric(label or symbol, "N/A")
        return np.nan
    pct = (last / prev - 1.0) * 100.0
    col.metric(label or symbol, f"{pct:+.2f}%", help=f"Change vs previous close • Last={last:,.2f}")
    return pct

# Indices / ETFs as proxies
kpi_for("SPY", kpi_cols[0], "S&P 500 (SPY)")
kpi_for("QQQ", kpi_cols[1], "Nasdaq 100 (QQQ)")
kpi_for("EFA", kpi_cols[2], "World ex-US (EFA)")

# FX headline (first selected pair)
headline_fx = fx_pairs[0] if fx_pairs else "USD/SEK"
fx_info = fx_latest_pair(headline_fx)
if pd.notnull(fx_info["rate"]):
    kpi_cols[3].metric(
        fx_info["pair"],
        f"{fx_info['rate']:.4f}",
        help=(f"Latest • {fx_info['date']}  |  1D: {fx_info['chg_1d']:+.2%}"
              if pd.notnull(fx_info["chg_1d"]) else f"Latest • {fx_info['date']}")
    )
else:
    kpi_cols[3].metric(fx_info["pair"], "N/A", help=f"Latest • {fx_info['date']}")

# Market mood (avg of watchlist 1D %)
mood_vals = []
for t in watchlist:
    prev, last = td_last_two(t)
    if prev and last:
        mood_vals.append(last / prev - 1.0)
avg_mood = np.mean(mood_vals) * 100 if mood_vals else np.nan
kpi_cols[4].metric("Market mood (avg 1D)", f"{avg_mood:+.2f}%" if pd.notnull(avg_mood) else "N/A")

st.divider()

# ----------------------
# FX SNAPSHOT CARDS
# -----------------------
st.subheader("FX Snapshot")
fx_cols = st.columns(min(4, max(1, len(fx_pairs))))
for i, pair in enumerate(fx_pairs):
    info = fx_latest_pair(pair)
    with fx_cols[i % len(fx_cols)]:
        if pd.notnull(info["rate"]):
            st.metric(
                info["pair"],
                f"{info['rate']:.4f}",
                help=(f"Date: {info['date']}  |  1D: {info['chg_1d']:+.2%}"
                      if pd.notnull(info["chg_1d"]) else f"Date: {info['date']}")
            )
        else:
            st.metric(info["pair"], "N/A", help=f"Date: {info['date']}")

        # Tiny sparkline using last ~15 business days
        try:
            base, quote = pair.split("/")
            end_dt = pd.to_datetime(info["date"], errors="coerce")
            if pd.isna(end_dt):
                end_dt = pd.Timestamp.today().normalize()
            start_dt = end_dt - pd.Timedelta(days=25)

            series_hist = fx.timeseries(start=start_dt.date(), end=end_dt.date(), base=base, symbols=quote)
            ser = _fx_series_from_timeseries(series_hist, quote)

            if not ser.empty:
                fig = px.line(x=ser.index, y=ser.values, height=120)
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No FX history")
        except Exception:
            st.caption("No FX history")

st.divider()

# -----------------------
# TOP MOVERS (1D)
# -----------------------
st.subheader("Top movers (1D)")
chg_rows = []
for t in watchlist:
    prev, last = td_last_two(t)
    if prev and last:
        chg_rows.append({"Ticker": t, "Change": last / prev - 1.0})

if chg_rows:
    movers = pd.DataFrame(chg_rows).sort_values("Change", ascending=False)
    show_top = min(10, len(movers))

    fig = px.bar(movers.head(show_top), x="Ticker", y="Change",
                 title=f"Top {show_top} gainers", text="Change")
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_tickformat=".1%", height=380)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(
        movers.tail(show_top).sort_values("Change"),
        x="Ticker", y="Change", title=f"Top {show_top} losers", text="Change"
    )
    fig2.update_traces(texttemplate="%{text:.2%}", textposition="outside", cliponaxis=False)
    fig2.update_layout(yaxis_tickformat=".1%", height=380)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Not enough data to compute 1D movers yet.")
