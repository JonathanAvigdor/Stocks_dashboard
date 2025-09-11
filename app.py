import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from typing import Optional, Tuple, List, Dict

from src.api.twelve_data import TwelveDataClient
from src.api.frankfurter import FrankfurterClient

if st.secrets.get("FETCH_ENABLED", "1") != "1":
    import streamlit as st
    st.warning("⏸️ Data fetching is disabled by server setting.")
    st.stop()


# -----------------------
# Page config & title
# -----------------------
st.set_page_config(page_title="Stocks & FX Dashboard", layout="wide")
st.title("📊 Stocks & FX Dashboard")
st.caption("Fast snapshot of markets: KPIs • Watchlist • FX • Top movers (optimized API usage)")

# -----------------------
# API keys / clients
# -----------------------
td_key = st.secrets.get("TWELVEDATA_API_KEY", "")
if not td_key or td_key == "YOUR_TWELVE_DATA_KEY":
    st.error("Please set TWELVEDATA_API_KEY in .streamlit/secrets.toml")
    st.stop()

td = TwelveDataClient(td_key)   # still available if you use it elsewhere
fx = FrankfurterClient()
TD_BASE = "https://api.twelvedata.com"

# -----------------------
# Refresh control (sidebar)
# -----------------------
with st.sidebar:
    st.header("Controls")

    if "refresh_counter" not in st.session_state:
        st.session_state["refresh_counter"] = 0
    if st.button("🔄 Refresh data"):
        st.session_state["refresh_counter"] += 1

    # Low API mode toggle (limits calls/features)
    low_api_mode = st.toggle("Low API mode (save credits)", value=True,
                             help="Limits watchlist size and disables sparklines to save API credits.")

    # Default watchlist
    DEFAULT_TICKERS = ["SPY", "QQQ", "EFA", "EWD", "AAPL", "MSFT", "NVDA"]
    watchlist = st.multiselect("Watchlist", DEFAULT_TICKERS, default=DEFAULT_TICKERS)

    # FX quick board
    fx_pairs = st.multiselect(
        "FX pairs",
        ["USD/EUR", "EUR/USD", "USD/SEK", "SEK/EUR", "GBP/USD", "EUR/SEK"],
        default=["USD/SEK", "EUR/USD", "EUR/SEK"]
    )

# Use this as a cache key so data only reloads when you click "Refresh"
REF = st.session_state["refresh_counter"]

# If saving credits, trim watchlist
if low_api_mode and len(watchlist) > 5:
    st.warning("Low API mode: limiting watchlist to first 5 tickers to save API credits.")
    watchlist = watchlist[:5]

# -----------------------
# Helpers (batched + cached)
# -----------------------
def _td_get(path: str, params: dict) -> dict:
    r = requests.get(f"{TD_BASE}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=1800, show_spinner=False)
def td_batch_last_two(symbols: List[str], api_key: str, _ref: int) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Fetch last 2 daily closes for many symbols in ONE call.
    Returns {symbol: (prev, last)}  ; missing -> (None, None)
    """
    if not symbols:
        return {}
    data = _td_get("time_series", {
        "symbol": ",".join(symbols),
        "interval": "1day",
        "outputsize": 2,
        "apikey": api_key,
        "format": "JSON",
    })

    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    # Single-symbol response shape
    if isinstance(data, dict) and "values" in data:
        vals = data.get("values", [])
        closes = [float(v["close"]) for v in reversed(vals)]
        sym = data.get("meta", {}).get("symbol", symbols[0])
        out[sym] = (None, None) if len(closes) < 2 else (closes[-2], closes[-1])
        return out

    # Multi-symbol shape: { "AAPL": {...}, "MSFT": {...}, ... }
    for sym in symbols:
        rec = data.get(sym, {})
        vals = rec.get("values", [])
        closes = [float(v["close"]) for v in reversed(vals)]
        out[sym] = (None, None) if len(closes) < 2 else (closes[-2], closes[-1])
    return out

@st.cache_data(ttl=1200, show_spinner=False)
def td_recent_series_batched(symbols: List[str], bars: int, api_key: str, _ref: int) -> Dict[str, pd.Series]:
    """
    Fetch recent closes for many symbols in ONE call. Returns {symbol: pd.Series}
    """
    if not symbols:
        return {}
    data = _td_get("time_series", {
        "symbol": ",".join(symbols),
        "interval": "1day",
        "outputsize": bars,
        "apikey": api_key,
        "format": "JSON",
    })

    def series_from_rec(rec: dict) -> pd.Series:
        vals = rec.get("values", [])
        if not vals:
            return pd.Series(dtype="float64")
        df = pd.DataFrame(vals)
        df["datetime"] = pd.to_datetime(df["datetime"])
        s = df.sort_values("datetime").set_index("datetime")["close"].astype(float)
        return s

    out: Dict[str, pd.Series] = {}
    if isinstance(data, dict) and "values" in data:  # single symbol case
        sym = data.get("meta", {}).get("symbol", symbols[0])
        out[sym] = series_from_rec(data)
        return out

    for sym in symbols:
        out[sym] = series_from_rec(data.get(sym, {}))
    return out

def _fx_series_from_timeseries(hist, quote: str) -> pd.Series:
    """Unify Frankfurter timeseries response (dict/DataFrame) to a pd.Series."""
    try:
        if isinstance(hist, dict) and "rates" in hist:
            ser = pd.Series({pd.to_datetime(k): float(v.get(quote)) for k, v in hist["rates"].items()}).sort_index()
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

@st.cache_data(ttl=900, show_spinner=False)
def fx_latest_pair(pair: str, _ref: int) -> dict:
    """
    pair format 'BASE/QUOTE' -> latest rate & approx 1D % change.
    Cached for 15 minutes and keyed by REF so manual Refresh updates it.
    """
    base, quote = pair.split("/")

    # Latest
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
        pass

    # 1D approx using small window
    chg_1d = np.nan
    try:
        end_dt = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(end_dt):
            end_dt = pd.Timestamp.utcnow().normalize()
        start_dt = end_dt - pd.Timedelta(days=5)
        hist = fx.timeseries(start=start_dt.date(), end=end_dt.date(), base=base, symbols=quote)
        ser = _fx_series_from_timeseries(hist, quote)
        if len(ser) >= 2 and ser.iloc[-2] != 0:
            chg_1d = float(ser.iloc[-1] / ser.iloc[-2] - 1.0)
    except Exception:
        pass

    return {"pair": pair, "rate": rate, "chg_1d": chg_1d, "date": str(date_str)}

# -----------------------
# KPI ROW (reusing batched data)
# -----------------------
kpi_cols = st.columns(5)

# ONE batched call for prev/last across the entire watchlist
last_two_map = td_batch_last_two(watchlist, td_key, REF)

def get_prev_last(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    return last_two_map.get(ticker, (None, None))

def kpi_for(symbol: str, col, label: Optional[str] = None):
    prev, last = get_prev_last(symbol)
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
fx_info = fx_latest_pair(headline_fx, REF)
kpi_cols[3].metric(
    fx_info["pair"],
    "N/A" if pd.isna(fx_info["rate"]) else f"{fx_info['rate']:.4f}",
    help=(f"Latest • {fx_info['date']}  |  1D: {fx_info['chg_1d']:+.2%}" if pd.notnull(fx_info["chg_1d"]) else f"Latest • {fx_info['date']}")
)

# Market mood (avg of watchlist 1D %), reusing last_two_map
mood_vals = []
for t in watchlist:
    prev, last = get_prev_last(t)
    if prev and last:
        mood_vals.append(last / prev - 1.0)
avg_mood = np.mean(mood_vals) * 100 if mood_vals else np.nan
kpi_cols[4].metric("Market mood (avg 1D)", f"{avg_mood:+.2f}%" if pd.notnull(avg_mood) else "N/A")

st.divider()

# -----------------------
# FX SNAPSHOT CARDS (cached)
# -----------------------
st.subheader("FX Snapshot")
fx_cols = st.columns(min(4, max(1, len(fx_pairs))))
for i, pair in enumerate(fx_pairs):
    info = fx_latest_pair(pair, REF)
    with fx_cols[i % len(fx_cols)]:
        if pd.notnull(info["rate"]):
            st.metric(
                info["pair"],
                f"{info['rate']:.4f}",
                help=(f"Date: {info['date']}  |  1D: {info['chg_1d']:+.2%}" if pd.notnull(info["chg_1d"]) else f"Date: {info['date']}")
            )
        else:
            st.metric(info["pair"], "N/A", help=f"Date: {info['date']}")

        # Tiny sparkline (kept lightweight)
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
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No FX history")
        except Exception:
            st.caption("No FX history")

st.divider()

# -----------------------
# WATCHLIST TABLE (batched lookbacks)
# -----------------------
st.subheader("Watchlist")
lookbacks = (1, 5, 21, 63)  # 1D, 1W, 1M, 3M approx

@st.cache_data(ttl=1800, show_spinner=False)
def td_lookbacks_pct_batched(symbols: List[str], lookbacks: tuple, api_key: str, _ref: int) -> Dict[str, Dict[str, float]]:
    max_lb = max(lookbacks) + 2
    series_map = td_recent_series_batched(symbols, bars=max(60, max_lb + 5), api_key=api_key, _ref=_ref)
    out: Dict[str, Dict[str, float]] = {}
    for sym, s in series_map.items():
        if s.empty or len(s) < max_lb:
            out[sym] = {"last": np.nan, **{f"{lb}d": np.nan for lb in lookbacks}}
            continue
        res = {"last": float(s.iloc[-1])}
        for lb in lookbacks:
            res[f"{lb}d"] = float(s.iloc[-1] / s.iloc[-lb-1] - 1.0) if s.iloc[-lb-1] != 0 else np.nan
        out[sym] = res
    return out

lb_map = td_lookbacks_pct_batched(watchlist, lookbacks, td_key, REF)

rows = []
for t in watchlist:
    stats = lb_map.get(t, {"last": np.nan, "1d": np.nan, "5d": np.nan, "21d": np.nan, "63d": np.nan})
    rows.append({
        "Ticker": t,
        "Last": stats["last"],
        "Δ 1D": stats["1d"],
        "Δ 5D": stats["5d"],
        "Δ 21D": stats["21d"],
        "Δ 63D": stats["63d"],
    })
watch_df = pd.DataFrame(rows)

fmt_df = watch_df.copy()
for c in ["Δ 1D", "Δ 5D", "Δ 21D", "Δ 63D"]:
    fmt_df[c] = fmt_df[c].apply(lambda x: "" if pd.isna(x) else f"{x:.2%}")
fmt_df["Last"] = fmt_df["Last"].apply(lambda v: "" if pd.isna(v) else f"{v:,.2f}")
st.dataframe(fmt_df, use_container_width=True)

# -----------------------
# SPARKLINES GRID (optional to save credits)
# -----------------------
if not low_api_mode:
    st.markdown("###### Sparklines (last ~25 trading days)")
    series_map = td_recent_series_batched(watchlist, bars=25, api_key=td_key, _ref=REF)
    spark_cols = st.columns(min(5, max(1, len(watchlist))))
    for i, t in enumerate(watchlist):
        s = series_map.get(t, pd.Series(dtype="float64"))
        with spark_cols[i % len(spark_cols)]:
            st.caption(t)
            if s.empty:
                st.write("no data")
            else:
                fig = px.line(x=s.index, y=s.values, height=120)
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False))
                st.plotly_chart(fig, use_container_width=True)

st.divider()

# -----------------------
# TOP MOVERS (1D) – reuse last_two_map
# -----------------------
st.subheader("Top movers (1D)")
chg_rows = []
for t in watchlist:
    prev, last = get_prev_last(t)
    if prev and last:
        chg_rows.append({"Ticker": t, "Change": last/prev - 1.0})

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
