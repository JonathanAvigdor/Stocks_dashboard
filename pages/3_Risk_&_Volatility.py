# pages/3_Risk_&_Volatility.py — yfinance version (tz-safe)

import streamlit as st
import pandas as pd
import plotly.express as px
import datetime as dt

from src import utils
from models.functions import get_price_series  # yfinance under the hood

st.title("⚡ Risk & Volatility")

# Optional kill switch (consistent with app.py)
if st.secrets.get("FETCH_ENABLED", "1") != "1":
    st.warning("⏸️ Data fetching is disabled by server setting.")
    st.stop()

# -------- Helpers --------
def _tz_naive_index(obj: pd.Series | pd.DataFrame):
    """
    Return a copy with a tz-naive DatetimeIndex (drop timezone).
    Converts via UTC when possible to avoid DST ambiguity.
    """
    if not isinstance(obj.index, pd.DatetimeIndex):
        return obj
    if obj.index.tz is None:
        return obj
    out = obj.copy()
    try:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    except Exception:
        out.index = out.index.tz_localize(None)
    return out

# -------- Refresh control (sidebar) --------
if "refresh_counter" not in st.session_state:
    st.session_state["refresh_counter"] = 0
if st.sidebar.button("🔄 Refresh data"):
    st.session_state["refresh_counter"] += 1

# -------- Inputs --------
tickers = st.multiselect(
    "Choose tickers",
    utils.DEFAULT_TICKERS,
    default=utils.DEFAULT_TICKERS[:3],
)
window = st.slider("Rolling window (days)", 10, 60, 30)

# Date range (default last 2 years)
date_range = st.date_input(
    "Date range",
    value=(dt.date.today() - dt.timedelta(days=365 * 2), dt.date.today()),
)
if isinstance(date_range, tuple):
    start_date, end_date = date_range
else:
    start_date, end_date = date_range - dt.timedelta(days=365), date_range

@st.cache_data(ttl=600, show_spinner=False)
def fetch_closes(symbol: str, bars: int = 5000, refresh_token: int = 0) -> pd.Series:
    """
    Fetch daily closes via yfinance wrapper.
    Ensures tz-naive index to avoid dtype comparison errors.
    'refresh_token' only busts cache when the refresh button is clicked.
    """
    s = get_price_series(symbol, api_key="", bars=bars)
    if s is None or isinstance(s, pd.Series) and s.empty:
        return pd.Series(dtype="float64")
    if not isinstance(s, pd.Series):
        return pd.Series(dtype="float64")
    s = _tz_naive_index(s)
    s.name = symbol
    return s

# -------- Load data --------
if not tickers:
    st.info("Pick at least one ticker.")
    st.stop()

closes: dict[str, pd.Series] = {}
for t in tickers:
    s = fetch_closes(t, bars=5000, refresh_token=st.session_state["refresh_counter"])
    if not s.empty:
        closes[t] = s

if not closes:
    st.info("No data loaded. Try different tickers.")
    st.stop()

# Align on common dates
panel = pd.concat(closes, axis=1).dropna()
panel = _tz_naive_index(panel)  # <- make sure the combined frame is tz-naive

# Filter by date range (both sides are tz-naive now)
start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)
panel = panel.loc[(panel.index >= start_ts) & (panel.index <= end_ts)]

if panel.empty:
    st.info("No data in the selected date range.")
    st.stop()

# -------- Returns & rolling volatility --------
returns = panel.pct_change().dropna()

rolling_vol = returns.rolling(window).std() * (252 ** 0.5)
st.plotly_chart(
    px.line(rolling_vol, title=f"Rolling Volatility ({window}D, annualized)"),
    use_container_width=True,
)

st.caption(
    "📊 **Volatility** shows how 'bumpy' the ride is: higher = bigger swings (riskier), lower = steadier. "
    f"This chart uses a {window}-day rolling window, scaled to annual levels."
)

# -------- Max drawdown --------
def max_drawdown(series: pd.Series) -> float:
    """Return the minimum drawdown as a decimal (e.g., -0.35 for -35%)."""
    if series is None or series.empty:
        return float("nan")
    roll_max = series.cummax()
    dd = (series - roll_max) / roll_max
    return dd.min() if not dd.empty else float("nan")

dd = {t: max_drawdown(panel[t]) for t in panel.columns}
dd_df = (
    pd.DataFrame({"ticker": list(dd.keys()), "max_drawdown": list(dd.values())})
    .sort_values("max_drawdown")  # most negative (worst) first
)

st.subheader("Max Drawdown (within selected date range)")
fmt_dd_df = dd_df.copy()
fmt_dd_df["max_drawdown"] = fmt_dd_df["max_drawdown"].apply(
    lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A"
)
st.dataframe(fmt_dd_df, use_container_width=True)

# -------- Volatility bar chart (grouped) --------
windows = [20, 60, 252, 1260]  # include ~5Y (1260 trading days)

vol_numeric: dict[str, dict[str, float | None]] = {}
for t in returns.columns:
    vols = {}
    for w in windows:
        if len(returns) >= w:
            vols[f"{w}D"] = returns[t].rolling(w).std().iloc[-1] * (252 ** 0.5)
        else:
            vols[f"{w}D"] = None
    vol_numeric[t] = vols

vol_num_df = pd.DataFrame(vol_numeric).T
vol_num_df.index.name = "Ticker"
vol_num_df = vol_num_df[[f"{w}D" for w in windows]]  # enforce column order

plot_df = (
    vol_num_df.reset_index()
    .melt(id_vars="Ticker", var_name="Window", value_name="Volatility")
    .dropna(subset=["Volatility"])
)

fig = px.bar(
    plot_df,
    x="Ticker",
    y="Volatility",
    color="Window",
    barmode="group",
    title="Annualized Volatility by Window (20D / 60D / 252D / 1260D)",
)
fig.update_layout(yaxis_tickformat=".1%")
st.plotly_chart(fig, use_container_width=True)

if len(returns) < 1260:
    st.caption("ℹ️ To see 5-year (1260D) volatility, expand the date range (≥ ~6–7 calendar years) and click **🔄 Refresh data**.")

# -------- Drawdown analysis (per-ticker) --------
st.subheader("Drawdown analysis")

def drawdown_stats(series: pd.Series):
    """
    Returns:
      dd_series: full drawdown time series (<= 0)
      mdd: max drawdown (float, negative)
      peak_date: date of peak before worst dd
      trough_date: date of trough (max dd)
      recovery_date: first date price recovered above that peak (if any)
    """
    if series is None or series.empty:
        return pd.Series(dtype="float64"), float("nan"), pd.NaT, pd.NaT, pd.NaT

    roll_max = series.cummax()
    dd_series = series / roll_max - 1.0

    mdd = dd_series.min()
    trough_date = dd_series.idxmin() if not dd_series.empty else pd.NaT
    peak_date = series.loc[:trough_date].idxmax() if pd.notna(trough_date) else pd.NaT

    recovery_date = pd.NaT
    if pd.notna(peak_date) and pd.notna(trough_date):
        post = series.loc[trough_date:]
        rec = post[post >= series.loc[peak_date]]
        if not rec.empty:
            recovery_date = rec.index[0]

    return dd_series, float(mdd), peak_date, trough_date, recovery_date

rows = []
dd_series_map = {}
for t in panel.columns:
    dd_series_t, mdd, peak_dt, trough_dt, rec_dt = drawdown_stats(panel[t])
    dd_series_map[t] = dd_series_t
    rows.append({
        "ticker": t,
        "max_drawdown": mdd,
        "peak_date": peak_dt,
        "trough_date": trough_dt,
        "recovery_date": rec_dt,
    })

dd_detail = pd.DataFrame(rows).sort_values("max_drawdown")  # worst first
fmt_detail = dd_detail.copy()
fmt_detail["max_drawdown"] = fmt_detail["max_drawdown"].apply(
    lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A"
)
st.dataframe(fmt_detail, use_container_width=True)

sel_ticker = st.selectbox("Drawdown chart — choose a ticker", options=list(panel.columns))
dd_sel, mdd_sel, pk_sel, tr_sel, rc_sel = drawdown_stats(panel[sel_ticker])

fig_dd = px.area(
    dd_sel,
    title=f"Drawdown: {sel_ticker} (Max: {mdd_sel:.2%})",
    labels={"value": "Drawdown", "index": "Date"},
)
fig_dd.update_traces(hovertemplate="Date=%{x|%Y-%m-%d}<br>Drawdown=%{y:.2%}<extra></extra>")
fig_dd.update_layout(yaxis_tickformat=".0%", yaxis_range=[-1, 0])

# vertical markers
shapes = []
annotations = []
def vline(x, text):
    return (
        dict(type="line", xref="x", x0=x, x1=x, yref="paper", y0=0, y1=1, line=dict(dash="dot")),
        dict(x=x, y=-0.02, xref="x", yref="paper", text=text, showarrow=False, yanchor="top"),
    )

if pd.notna(pk_sel):
    s, a = vline(pk_sel, "Peak"); shapes.append(s); annotations.append(a)
if pd.notna(tr_sel):
    s, a = vline(tr_sel, "Trough"); shapes.append(s); annotations.append(a)
if pd.notna(rc_sel):
    s, a = vline(rc_sel, "Recovery"); shapes.append(s); annotations.append(a)

fig_dd.update_layout(shapes=shapes, annotations=annotations)
st.plotly_chart(fig_dd, use_container_width=True)

st.caption(
    "📉 **Drawdown** shows how far an investment has fallen from its peak. "
    "It highlights worst-case drops (pain points) and how long recovery took."
)
