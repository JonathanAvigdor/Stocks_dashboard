# pages/Stock_Explorer.py — yfinance version
import streamlit as st
import pandas as pd
import datetime as dt

from src import utils, charts
from models.functions import get_price_series  # uses yfinance under the hood

st.title("📈 Stock Explorer")

# Optional kill switch to match main app behavior
if st.secrets.get("FETCH_ENABLED", "1") != "1":
    st.warning("⏸️ Data fetching is disabled by server setting.")
    st.stop()

# ---------- Refresh (sidebar) ----------
if "refresh_counter" not in st.session_state:
    st.session_state["refresh_counter"] = 0
with st.sidebar:
    if st.button("🔄 Refresh data"):
        st.session_state["refresh_counter"] += 1

# ---------- Inputs ----------
symbol = st.text_input(
    "Symbol",
    value="AAPL",
    placeholder="e.g., AAPL, SPY, VOLV-B:XSTO, TEVA:XTAE",
    help="US: AAPL, MSFT, SPY…  Non-US often needs SYMBOL:MIC (e.g., VOLV-B:XSTO, TEVA:XTAE)."
)

hist = st.selectbox("History", ["Last ~120 days (compact)", "Full (~500+)"], index=0)
bars = 120 if "compact" in hist else 5000  # yfinance is cheap, let’s give plenty for full

# ---------- Fetch ----------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_df(sym: str, bars: int, _ref: int) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: time, close
    Uses models.functions.get_price_series (yfinance).
    """
    s = get_price_series(sym, api_key="", bars=bars)
    if s is None or s.empty:
        return pd.DataFrame(columns=["time", "close"])
    df = pd.DataFrame({"time": pd.to_datetime(s.index), "close": pd.to_numeric(s.values, errors="coerce")})
    df = df.dropna(subset=["close"]).sort_values("time")
    return df

if not symbol.strip():
    st.info("Enter a symbol to begin.")
    st.stop()

df = fetch_df(symbol.strip(), bars=bars, _ref=st.session_state["refresh_counter"])

if df.empty:
    st.warning(
        "No data returned. Try a different symbol.\n\n"
        "Tip: For non-US listings use SYMBOL:MIC (e.g., **VOLV-B:XSTO**, **TEVA:XTAE**)."
    )
    st.stop()

# ---------- Moving averages & Chart ----------
df_ma = utils.compute_moving_averages(df)  # expects columns: time, close
st.plotly_chart(
    charts.line(df_ma, x="time", y="close", title=f"{symbol} — Close"),
    use_container_width=True
)

# ---------- Data preview ----------
with st.expander("Data (last 30 rows)"):
    st.dataframe(df_ma.tail(30), use_container_width=True)
