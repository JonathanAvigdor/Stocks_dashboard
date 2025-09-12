# pages/Overview.py — yfinance snapshot + Frankfurter FX
import streamlit as st
import pandas as pd
import yfinance as yf

from src.api.frankfurter import FrankfurterClient
from src.adapters.yahoo_map import td_to_yahoo  # TD-style -> Yahoo symbols (e.g., VOLV-B:XSTO -> VOLV-B.ST)

# ---------------
# Page setup
# ---------------
st.set_page_config(page_title="Stocks & FX Dashboard", layout="wide")
st.title("📊 Stocks & FX Dashboard")
st.subheader("Overview")

# Optional kill switch (same behavior as main app)
if st.secrets.get("FETCH_ENABLED", "1") != "1":
    st.warning("⏸️ Data fetching is disabled by server setting.")
    st.stop()

fx = FrankfurterClient()

# ===============================
# LEFT: Index / Stock Snapshot
# ===============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Index / Stock Snapshot (Δ% vs previous close)")
    default_tickers = ["SPY", "QQQ", "EFA", "EWD", "AAPL", "MSFT", "NVDA"]
    selected = st.multiselect("Tickers", default_tickers, default=default_tickers[:5])

    @st.cache_data(ttl=120, show_spinner=False)
    def pct_changes_batch(symbols: list[str]) -> pd.DataFrame:
        """
        One yfinance call for all tickers:
        Returns DataFrame: columns=['symbol','change_%'] where change_% is vs previous close.
        """
        if not symbols:
            return pd.DataFrame(columns=["symbol", "change_%"])

        # Map TD-style symbols to Yahoo tickers
        mapping = {sym: td_to_yahoo(sym) for sym in symbols}
        ysyms = list(mapping.values())

        df = yf.download(
            tickers=ysyms,
            period="7d",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        rows = []

        # Handle both shapes: MultiIndex (many tickers) or single-index (one ticker)
        if isinstance(df.columns, pd.MultiIndex):
            for td_sym, ysym in mapping.items():
                try:
                    s = df[ysym]["Adj Close"].dropna()
                except Exception:
                    try:
                        s = df[ysym]["Close"].dropna()
                    except Exception:
                        s = pd.Series(dtype="float64")
                if len(s) >= 2 and s.iloc[-2] != 0:
                    chg = (float(s.iloc[-1]) / float(s.iloc[-2]) - 1.0) * 100.0
                    rows.append({"symbol": td_sym, "change_%": round(chg, 2)})
                else:
                    rows.append({"symbol": td_sym, "change_%": None})
        else:
            # Single ticker shape
            td_sym = symbols[0]
            s = df.get("Adj Close")
            if s is None:
                s = df.get("Close")
            s = (s or pd.Series(dtype="float64")).dropna()
            if len(s) >= 2 and s.iloc[-2] != 0:
                chg = (float(s.iloc[-1]) / float(s.iloc[-2]) - 1.0) * 100.0
                rows.append({"symbol": td_sym, "change_%": round(chg, 2)})
            else:
                rows.append({"symbol": td_sym, "change_%": None})

        # Preserve input order
        out = pd.DataFrame(rows)
        out["symbol"] = pd.Categorical(out["symbol"], categories=symbols, ordered=True)
        return out.sort_values("symbol").reset_index(drop=True)

    if selected:
        df_changes = pct_changes_batch(selected)
        st.dataframe(df_changes, use_container_width=True)
    else:
        st.info("Pick at least one ticker.")

# =================
# RIGHT: FX Snapshot
# =================
with col2:
    st.subheader("FX Snapshot")
    base = st.selectbox("Base", ["USD", "EUR", "SEK", "GBP"], index=2)   # default SEK
    quote = st.selectbox("Quote", ["USD", "EUR", "SEK", "GBP"], index=1)  # default EUR

    if base == quote:
        st.warning("Choose different currencies.")
    else:
        latest = fx.latest(base=base, symbols=quote)
        try:
            rate = latest["rates"][quote]
            date = latest.get("date", "")
            st.metric(f"{base}/{quote}", f"{rate:.4f}" if isinstance(rate, (int, float)) else rate, help=f"Date: {date}")
        except Exception:
            st.error("Could not load FX rate right now.")
