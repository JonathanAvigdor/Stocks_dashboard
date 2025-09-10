import streamlit as st
from src.api.twelve_data import TwelveDataClient
from src import utils, charts

st.title("📈 Stock Explorer")

# --- API key (Twelve Data) ---
api_key = st.secrets.get("TWELVEDATA_API_KEY", "")
if not api_key or api_key == "YOUR_KEY_HERE":
    st.error("Please set TWELVEDATA_API_KEY in .streamlit/secrets.toml")
    st.stop()

td = TwelveDataClient(api_key)

# --- Inputs ---
symbol = st.text_input("Symbol (e.g., AAPL, MSFT, SPY)", value="AAPL")
hist = st.selectbox("History", ["Last ~100 days (compact)", "Full (~500)"], index=0)
outputsize = 100 if "compact" in hist else 500

# --- Fetch & show ---
df = td.daily(symbol, interval="1day", outputsize=outputsize)

if df.empty:
    st.warning("No data returned. Try AAPL / MSFT / SPY, or check your Twelve Data key.")
    st.stop()

df_ma = utils.compute_moving_averages(df)
st.plotly_chart(
    charts.line(df_ma, x="time", y="close", title=f"{symbol} — Close"),
    use_container_width=True
)

with st.expander("Data (last 30 rows)"):
    st.dataframe(df_ma.tail(30), use_container_width=True)
