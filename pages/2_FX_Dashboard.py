import streamlit as st
from src.api.frankfurter import FrankfurterClient
from src import charts

st.title("💱 FX Dashboard")
fx = FrankfurterClient()

base = st.selectbox("Base", ["USD","EUR","SEK","GBP"], index=2)
quote = st.selectbox("Quote", ["USD","EUR","SEK","GBP"], index=1)
start = st.date_input("Start")
end = st.date_input("End")

if base == quote:
    st.warning("Choose different currencies.")
else:
    if start and end and start < end:
        df = fx.timeseries(start.isoformat(), end.isoformat(), base=base, symbols=quote)
        st.plotly_chart(charts.line(df, x="time", y="rate", title=f"{base}/{quote}"), use_container_width=True)
    else:
        latest = fx.latest(base=base, symbols=quote)
        st.metric(f"{base}/{quote}", latest["rates"][quote], help=f"Date: {latest['date']}")
