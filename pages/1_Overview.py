import streamlit as st
import pandas as pd
from src.api.twelve_data import TwelveDataClient
from src.api.frankfurter import FrankfurterClient


st.set_page_config(page_title="Stocks & FX Dashboard", layout="wide")
st.title("📊 Stocks & FX Dashboard")

st.subheader("Overview")

# --- Keys ---
td_key = st.secrets.get("TWELVEDATA_API_KEY", "")
if not td_key or td_key == "YOUR_TWELVE_DATA_KEY":
    st.error("Please set TWELVEDATA_API_KEY in .streamlit/secrets.toml")
    st.stop()

td = TwelveDataClient(td_key)
fx = FrankfurterClient()

col1, col2 = st.columns(2)

# ---------- LEFT: STOCK / INDEX SNAPSHOT ----------
with col1:
    st.subheader("Index / Stock Snapshot (Δ% vs previous close)")
    # ETFs as index proxies + a couple of large caps
    default_tickers = ["SPY", "QQQ", "EFA", "EWD", "AAPL", "MSFT", "NVDA"]
    selected = st.multiselect("Tickers", default_tickers, default=default_tickers[:5])

    @st.cache_data(ttl=120)
    def pct_change_last_close(symbol: str, api_key: str) -> float | None:
        # Get last 2 daily bars and compute % change
        client = TwelveDataClient(api_key)
        df = client.daily(symbol, interval="1day", outputsize=2)
        if df is None or df.empty or len(df) < 2:
            return None
        last2 = df.tail(2)["close"].to_list()
        prev, last = last2[0], last2[1]
        if prev is None or prev == 0:
            return None
        return (last / prev - 1.0) * 100.0

    rows = []
    for t in selected:
        chg = pct_change_last_close(t, td_key)
        rows.append({"symbol": t, "change_%": 0.0 if chg is None else round(chg, 2)})

    df_changes = pd.DataFrame(rows)
    st.dataframe(df_changes, use_container_width=True)

# ---------- RIGHT: FX SNAPSHOT ----------
with col2:
    st.subheader("FX Snapshot")
    base = st.selectbox("Base", ["USD", "EUR", "SEK", "GBP"], index=2)   # default SEK
    quote = st.selectbox("Quote", ["USD", "EUR", "SEK", "GBP"], index=1)  # default EUR
    if base == quote:
        st.warning("Choose different currencies.")
    else:
        latest = fx.latest(base=base, symbols=quote)
        st.metric(f"{base}/{quote}", latest["rates"][quote], help=f"Date: {latest['date']}")
