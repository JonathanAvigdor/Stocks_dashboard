# app.py — Stocks & FX Dashboard (yfinance prices + global symbol search)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from typing import Optional, Tuple, List, Dict

# FX client
from src.api.frankfurter import FrankfurterClient

# Symbol search (cached TD reference endpoints; NOT used for pricing)
from src.api.symbols import (
    EXCHANGE_MAP,
    search_symbols,
    format_suggestion_row,
    add_to_watchlist,
    normalize_symbol,
    debounce_ok,
    ensure_session_state,
    _mic_of,  # normalize venue names to MIC (e.g., "TASE" -> XTAE)
)

# =========================================================
# Kill switch (flip FETCH_ENABLED to "0" in Streamlit Cloud)
# =========================================================
if st.secrets.get("FETCH_ENABLED", "1") != "1":
    st.warning("⏸️ Data fetching is disabled by server setting.")
    st.stop()

# ================ Page config & header ================
st.set_page_config(page_title="Stocks & FX Dashboard", layout="wide")
st.title("📊 Stocks & FX Dashboard")
st.caption("Fast snapshot of markets: KPIs • Watchlist • FX • Top movers (single-call optimized)")

# ========================= Init state & defaults =========================
DEFAULT_TICKERS = ["SPY", "QQQ", "EFA", "EWD", "AAPL", "MSFT", "NVDA"]
ensure_session_state()
if "watchlist_seeded" not in st.session_state:
    st.session_state.watchlist = list(DEFAULT_TICKERS)
    st.session_state.watchlist_seeded = True

fx = FrankfurterClient()

# ================= Sidebar: controls + global search =================
with st.sidebar:
    st.header("Controls")

    if "refresh_counter" not in st.session_state:
        st.session_state["refresh_counter"] = 0
    if st.button("🔄 Refresh data"):
        st.session_state["refresh_counter"] += 1

    low_api_mode = st.toggle(
        "Low API mode (save credits)",
        value=True,
        help="Limits watchlist size and disables sparklines to save bandwidth."
    )

    st.markdown("---")
    st.subheader("🔎 Add symbols")

    exchange_display = st.selectbox(
        "Exchange",
        options=list(EXCHANGE_MAP.keys()),
        index=0,
        help="US works with plain symbols (AAPL, MSFT). For Sweden/Israel we add the MIC (XSTO, XTAE).",
    )
    exchange_hint = EXCHANGE_MAP[exchange_display]

    query = st.text_input(
        "Search by symbol or name…",
        value="",
        placeholder="e.g., AAPL, VOLV-B, TEVA",
        help="Type 1–4 letters for suggestions. Click a suggestion to add.",
        key="__symbol_query",
    )

    # Free-text add (normalize + validate)
    c_free1, c_free2 = st.columns([0.65, 0.35])
    with c_free1:
        free_text = st.text_input(
            "Add typed symbol (optional)",
            value="",
            placeholder="e.g., VOLV-B:XSTO",
            label_visibility="collapsed",
            key="__typed_symbol",
        )
    with c_free2:
        if st.button("Add", help="Add the typed symbol as-is (we’ll normalize/validate)."):
            sym_to_add = free_text.strip()
            if sym_to_add:
                ok, err, norm = add_to_watchlist(sym_to_add, exchange_hint=exchange_hint)
                if ok:
                    st.toast(f"Added {norm}")
                    st.session_state.__typed_symbol = ""  # clear input
                else:
                    st.toast(err or "Could not add.", icon="⚠️")

    # Debounced suggestions
    if query.strip():
        if debounce_ok(500):
            suggestions = search_symbols(prefix=query.strip(), exchange=exchange_hint, country=None, limit=20)
            if suggestions:
                st.caption("Suggestions")
                for i, s in enumerate(suggestions):
                    label = format_suggestion_row(s)
                    base = s.get("symbol", "")
                    venue = s.get("exchange", "")
                    norm = normalize_symbol(base, exchange_hint=(_mic_of(venue) if venue else exchange_hint))

                    c1, c2 = st.columns([0.8, 0.2])
                    with c1:
                        st.write(label)
                    with c2:
                        if st.button("Add", key=f"add_{norm}_{i}"):
                            ok, err, _ = add_to_watchlist(norm, exchange_hint=exchange_hint)
                            if ok:
                                st.toast(f"Added {norm}")
                            else:
                                st.toast(err or "Could not add.", icon="⚠️")
            else:
                st.info("No matches. Try another symbol or switch exchange.")

    st.markdown("---")
    st.subheader("📌 Watchlist")
    if st.session_state.watchlist:
        to_keep = st.multiselect(
            "Selected symbols (uncheck to remove)",
            options=st.session_state.watchlist,
            default=st.session_state.watchlist,
        )
        new_list = list(dict.fromkeys(to_keep))  # keep order, dedupe
        if new_list != st.session_state.watchlist:
            st.session_state.watchlist = new_list
            st.toast("Watchlist updated")
    else:
        st.caption("Empty. Try adding AAPL, VOLV-B:XSTO, or TEVA:XTAE.")

    st.markdown("---")
    fx_pairs = st.multiselect(
        "FX pairs",
        ["USD/EUR", "EUR/USD", "USD/SEK", "SEK/EUR", "GBP/USD", "EUR/SEK"],
        default=["USD/SEK", "EUR/USD", "EUR/SEK"]
    )

# ========================================================
# Cache key so price data only reloads on manual refresh
# ========================================================
REF = st.session_state["refresh_counter"]

# Limit watchlist in low API mode
watchlist = list(st.session_state.watchlist)
if low_api_mode and len(watchlist) > 5:
    st.warning("Low API mode: limiting watchlist to first 5 tickers to save bandwidth.")
    watchlist = watchlist[:5]

# ============================================
# yfinance helper (ONE batched fetch per refresh)
# ============================================
@st.cache_data(ttl=900, show_spinner=False)
def yf_series_all(symbols: List[str], bars: int, _ref: int) -> Dict[str, pd.Series]:
    """
    Batch download daily adjusted closes for all symbols in one yfinance call.
    Uses a conservative period (no 'max') to avoid Yahoo quirks on some indices.
    """
    if not symbols:
        return {}

    from src.adapters.yahoo_map import td_to_yahoo
    mapping = {sym: td_to_yahoo(sym) for sym in symbols}
    ylist = list(mapping.values())

    # choose a small-but-safe period based on bars requested
    if bars <= 90:
        period = "1y"
    elif bars <= 252:
        period = "2y"
    elif bars <= 1260:
        period = "5y"
    else:
        period = "10y"

    df = yf.download(
        tickers=ylist,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    out: Dict[str, pd.Series] = {}

    # Multi-ticker -> MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        for td_sym, ysym in mapping.items():
            try:
                s = df[ysym]["Adj Close"] if ("Adj Close" in df[ysym].columns) else df[ysym].get("Close")
                if s is None or s.empty:
                    out[td_sym] = pd.Series(dtype="float64")
                    continue
                s = s.dropna()
                if bars:
                    s = s.tail(bars)
                s.index = pd.to_datetime(s.index)
                out[td_sym] = s.astype(float)
            except Exception:
                out[td_sym] = pd.Series(dtype="float64")
    else:
        # Single-ticker shape
        td_sym = symbols[0]
        s = df["Adj Close"] if "Adj Close" in df else df.get("Close")
        s = (s.dropna().tail(bars) if s is not None else pd.Series(dtype="float64"))
        if not s.empty:
            s.index = pd.to_datetime(s.index)
            out[td_sym] = s.astype(float)
        else:
            out[td_sym] = pd.Series(dtype="float64")

    return out


# ============================ Small helpers ============================
def last_two_from_series(s: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    if s is None or s.empty or len(s) < 2:
        return None, None
    return float(s.iloc[-2]), float(s.iloc[-1])

def lookbacks_from_series(s: pd.Series, lookbacks=(1, 5, 21, 63)) -> dict:
    if s is None or s.empty:
        return {"last": np.nan, **{f"{lb}d": np.nan for lb in lookbacks}}
    out = {"last": float(s.iloc[-1])}
    for lb in lookbacks:
        if len(s) > lb and s.iloc[-lb-1] != 0:
            out[f"{lb}d"] = float(s.iloc[-1] / s.iloc[-lb-1] - 1.0)
        else:
            out[f"{lb}d"] = np.nan
    return out

# ======================= Frankfurter FX helpers =======================
def _fx_series_from_timeseries(hist, quote: str) -> pd.Series:
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
    base, quote = pair.split("/")
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

# ====================================== Decide data needs & fetch ======================================
lookbacks = (1, 5, 21, 63)
kpi_syms = ["SPY", "QQQ", "EFA"]
all_syms = list(dict.fromkeys([*(watchlist or []), *kpi_syms]))  # unique, order-preserving

need_bars = max(60, max(lookbacks) + 5)   # enough for lookbacks
if not low_api_mode:
    need_bars = max(need_bars, 25)        # sparklines depth

if not all_syms:
    st.warning("Your watchlist is empty. Add symbols from the sidebar to start.")
    st.stop()

# ONE yfinance call for all symbols:
series_map = yf_series_all(all_syms, bars=need_bars, _ref=REF)

def get_prev_last(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    return last_two_from_series(series_map.get(ticker, pd.Series(dtype="float64")))

# ========================== KPI row ==========================
kpi_cols = st.columns(5)

def kpi_for(symbol: str, col, label: Optional[str] = None):
    s = series_map.get(symbol, pd.Series(dtype="float64"))
    prev, last = last_two_from_series(s)
    if prev is None or last is None:
        col.metric(label or symbol, "N/A")
        return np.nan
    pct = (last / prev - 1.0) * 100.0
    last_dt = (s.index[-1].date().isoformat() if len(s) else "")
    prev_dt = (s.index[-2].date().isoformat() if len(s) > 1 else "")
    col.metric(
        label or symbol,
        f"{pct:+.2f}%",
        help=f"Change vs previous close • Last={last:,.2f} ({last_dt}) • Prev={prev:,.2f} ({prev_dt})"
    )
    return pct

kpi_for("SPY", kpi_cols[0], "S&P 500 (SPY)")
kpi_for("QQQ", kpi_cols[1], "Nasdaq 100 (QQQ)")
kpi_for("EFA", kpi_cols[2], "World ex-US (EFA)")

headline_fx = fx_pairs[0] if fx_pairs else "USD/SEK"
fx_info = fx_latest_pair(headline_fx, REF)
kpi_cols[3].metric(
    fx_info["pair"],
    "N/A" if pd.isna(fx_info["rate"]) else f"{fx_info['rate']:.4f}",
    help=(f"Latest • {fx_info['date']}  |  1D: {fx_info['chg_1d']:+.2%}" if pd.notnull(fx_info["chg_1d"]) else f"Latest • {fx_info['date']}")
)

mood_vals = []
for t in watchlist:
    prev, last = get_prev_last(t)
    if prev and last:
        mood_vals.append(last / prev - 1.0)
avg_mood = np.mean(mood_vals) * 100 if mood_vals else np.nan
kpi_cols[4].metric("Market mood (avg 1D)", f"{avg_mood:+.2f}%" if pd.notnull(avg_mood) else "N/A")

st.divider()

# ====================== FX Snapshot ======================
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

        # Tiny FX sparkline (Frankfurter)
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

# ====================== Watchlist table ======================
st.subheader("Watchlist")
rows = []
for t in watchlist:
    stats = lookbacks_from_series(series_map.get(t, pd.Series(dtype="float64")), lookbacks=lookbacks)
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

# ====================== Sparklines ======================
if not low_api_mode and watchlist:
    st.markdown("###### Sparklines (last ~25 trading days)")
    spark_cols = st.columns(min(5, max(1, len(watchlist))))
    for i, t in enumerate(watchlist):
        s = series_map.get(t, pd.Series(dtype="float64"))
        if not s.empty and len(s) > 25:
            s = s.iloc[-25:]
        with spark_cols[i % len(spark_cols)]:
            st.caption(t)
            if s.empty:
                st.write("no data")
            else:
                fig = px.line(x=s.index, y=s.values, height=120)
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(visible=False), yaxis=dict(visible=False))
                st.plotly_chart(fig, use_container_width=True)

st.divider()

# ====================== Top movers (1D) ======================
st.subheader("Top movers (1D)")
chg_rows = []
for t in watchlist:
    prev, last = get_prev_last(t)
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
