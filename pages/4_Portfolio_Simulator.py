import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px

from src import utils
from models.functions import (
    get_price_series,   # yfinance under the hood
    compute_returns,
    normalize_weights,
    simulate_portfolio,
)

# === Global symbol search helpers (reference-only; not used for pricing)
from src.api.symbols import (
    EXCHANGE_MAP,
    search_symbols,
    format_suggestion_row,
    add_to_watchlist,
    normalize_symbol,
    debounce_ok,
    ensure_session_state,
    _mic_of,
)

# -----------------------
# Helpers
# -----------------------
def _tz_naive_index(obj):
    """Return a copy with DatetimeIndex converted to tz-naive (drops timezone)."""
    if isinstance(obj.index, pd.DatetimeIndex) and obj.index.tz is not None:
        obj = obj.copy()
        obj.index = obj.index.tz_localize(None)
    return obj

# -----------------------
# Page title
# -----------------------
st.title("🧪 Portfolio Simulator")

# -----------------------
# Refresh control (sidebar)
# -----------------------
if "refresh_counter" not in st.session_state:
    st.session_state["refresh_counter"] = 0
with st.sidebar:
    if st.button("🔄 Refresh data"):
        st.session_state["refresh_counter"] += 1

# -----------------------
# Seed & global watchlist (once)
# -----------------------
ensure_session_state()
if "sim_watchlist_seeded" not in st.session_state:
    st.session_state.watchlist = list(utils.DEFAULT_TICKERS)
    st.session_state.sim_watchlist_seeded = True

# -----------------------
# Sidebar: Global symbol search (US + XSTO + XTAE)
# -----------------------
with st.sidebar:
    st.header("Add symbols (global)")
    exchange_display = st.selectbox(
        "Exchange",
        options=list(EXCHANGE_MAP.keys()),
        index=0,
        help="US works with plain symbols (AAPL, MSFT). For Sweden/Israel we add the MIC (:XSTO, :XTAE).",
        key="sim_exch_select",
    )
    exchange_hint = EXCHANGE_MAP[exchange_display]

    query = st.text_input(
        "Search by symbol or name…",
        value="",
        placeholder="e.g., AAPL, VOLV-B, TEVA",
        help="Type 1–4 letters for suggestions. Click a suggestion to add.",
        key="sim_query",
    )

    # Free-text add (power users)
    c1, c2 = st.columns([0.65, 0.35])
    with c1:
        typed = st.text_input(
            "Add typed symbol (optional)",
            value="",
            placeholder="e.g., VOLV-B:XSTO",
            label_visibility="collapsed",
            key="sim_typed",
        )
    with c2:
        if st.button("Add", help="Add typed symbol; we'll normalize/validate."):
            raw = typed.strip()
            if raw:
                ok, err, norm = add_to_watchlist(raw, exchange_hint=exchange_hint)
                if ok:
                    st.toast(f"Added {norm}")
                    st.session_state.sim_typed = ""
                else:
                    st.toast(err or "Could not add.", icon="⚠️")

    # Typeahead suggestions (debounced)
    if query.strip():
        if debounce_ok(500):
            sugs = search_symbols(prefix=query.strip(), exchange=exchange_hint, country=None, limit=20)
            if sugs:
                st.caption("Suggestions")
                for i, s in enumerate(sugs):
                    label = format_suggestion_row(s)
                    base = s.get("symbol", "")
                    venue_raw = s.get("exchange", "")
                    norm = normalize_symbol(
                        base,
                        exchange_hint=(_mic_of(venue_raw) if venue_raw else exchange_hint)
                    )
                    c3, c4 = st.columns([0.8, 0.2])
                    with c3:
                        st.write(label)
                    with c4:
                        if st.button("Add", key=f"sim_add_{norm}_{i}"):
                            ok, err, _ = add_to_watchlist(norm, exchange_hint=exchange_hint)
                            if ok:
                                st.toast(f"Added {norm}")
                            else:
                                st.toast(err or "Could not add.", icon="⚠️")
            else:
                st.info("No matches. Try another symbol or switch exchange.")

# -----------------------
# Sidebar Inputs (use the global watchlist as universe)
# -----------------------
with st.sidebar:
    st.header("Inputs")

    # Universe = your global watchlist
    if st.session_state.watchlist:
        tickers = st.multiselect(
            "Choose tickers / indexes",
            options=st.session_state.watchlist,
            default=st.session_state.watchlist[:3],
            key="sim_tickers_select",
        )
    else:
        tickers = st.multiselect(
            "Choose tickers / indexes",
            options=[],
            default=[],
            key="sim_tickers_select_empty",
        )

    # Date range
    date_range = st.date_input(
        "Date range",
        value=(dt.date.today() - dt.timedelta(days=365 * 7), dt.date.today())
    )
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date, end_date = date_range - dt.timedelta(days=365 * 5), date_range

    # Contributions
    initial_investment = st.number_input("Initial investment", min_value=0, value=10000, step=500)
    monthly_contribution = st.number_input("Monthly contribution (DCA)", min_value=0, value=500, step=100)

    # Rebalancing (default Yearly)
    rebal_freq = st.selectbox(
        "Rebalancing",
        options=["None", "Monthly", "Quarterly", "Yearly"],
        index=3
    )

    # Return type
    return_type = st.selectbox("Return type", options=["Simple", "Log"], index=0)

# -----------------------
# Cached fetch wrapper (yfinance via get_price_series)
# -----------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_closes(symbol: str, bars: int = 5000, refresh_token: int = 0) -> pd.Series:
    """
    Uses models.functions.get_price_series (yfinance under the hood).
    'refresh_token' just busts the cache on manual refresh.
    Ensures tz-naive index to avoid comparison errors.
    """
    s = get_price_series(symbol, api_key="", bars=bars)
    if isinstance(s, pd.Series) and not s.empty:
        s = _tz_naive_index(s)
        s.name = s.name or symbol
        return s
    return pd.Series(dtype="float64", name=symbol)

# -----------------------
# Load & align data
# -----------------------
if len(tickers) == 0:
    st.info("Select at least one ticker from your watchlist in the sidebar.")
    st.stop()

closes = {}
failed = []
for t in tickers:
    s = fetch_closes(t, bars=5000, refresh_token=st.session_state["refresh_counter"])
    if not s.empty:
        s.name = t  # keep the user's TD-style label (e.g., VOLV-B:XSTO)
        closes[t] = s
    else:
        failed.append(t)

if failed:
    st.warning(
        "No data for: " + ", ".join(failed) +
        ". For non-US listings use SYMBOL:MIC (e.g., VOLV-B:XSTO, TEVA:XTAE) "
        "or their Yahoo forms (e.g., VOLV-B.ST, TEVA.TA)."
    )
if not closes:
    st.warning("No data loaded for the selected tickers.")
    st.stop()

prices = pd.concat(closes, axis=1).dropna()
prices = _tz_naive_index(prices)  # ensure tz-naive before date filtering

# Date filter (now both sides are tz-naive)
start = pd.Timestamp(start_date)
end = pd.Timestamp(end_date)
prices = prices.loc[(prices.index >= start) & (prices.index <= end)]

if prices.empty:
    st.warning("No price data in the selected date range.")
    st.stop()

# -----------------------
# Returns (for correlation tab etc.)
# -----------------------
asset_rets = compute_returns(prices, kind=return_type)

# -----------------------
# Weights editor (auto-normalized)
# -----------------------
st.subheader("Portfolio weights")

default_weights = (np.ones(len(prices.columns)) / len(prices.columns)) * 100.0
weights_df = pd.DataFrame({"Ticker": list(prices.columns), "Weight (%)": default_weights})

edited = st.data_editor(
    weights_df,
    num_rows="fixed",
    key="weights_editor",
    column_config={
        "Weight (%)": st.column_config.NumberColumn(
            "Weight (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.2f"
        )
    }
)

norm = normalize_weights(edited["Weight (%)"])
weights = pd.Series(norm.values, index=edited["Ticker"].tolist(), name="Weight (normalized)")
st.caption(f"Normalized weights (sum = 100%): {', '.join([f'{t}: {wt*100:.2f}%' for t, wt in weights.items()])}")

# Filter weights to available price columns to avoid KeyError
available = set(prices.columns)
missing = [t for t in weights.index if t not in available]
if missing:
    st.warning(f"Dropped (no data): {', '.join(missing)}")
weights = weights.loc[[t for t in weights.index if t in available]]
if weights.empty:
    st.error("None of the selected tickers returned data. Try different tickers or extend the date range.")
    st.stop()
weights = weights / weights.sum()

# -----------------------
# Simulation (historical backtest to today)
# -----------------------
sim = simulate_portfolio(
    prices_df=prices[weights.index],   # ensure columns match weights order
    target_w=weights,
    initial=initial_investment,
    monthly=monthly_contribution,
    rebalancing=rebal_freq
)

value = sim["value"]
flows = sim["flows"]
twr = sim["twr_returns"]
alloc = sim["alloc"]

# KPIs
total_contrib = float(flows.sum())
final_value = float(value.iloc[-1])
net_profit = final_value - total_contrib

valid_twr = (1 + twr.fillna(0)).cumprod()
twr_total_return = float(valid_twr.iloc[-1] - 1.0)

n_days = (value.index[-1] - value.index[0]).days
years = max(n_days / 365.25, 1e-9)
twr_cagr = float((1 + twr_total_return) ** (1 / years) - 1)

ann_vol = float(twr.std(ddof=0) * np.sqrt(252))
dd_series = value / value.cummax() - 1.0
max_dd = float(dd_series.min())

# -----------------------
# Tabs (incl. Forecast)
# -----------------------
tab_overview, tab_performance, tab_risk, tab_corr, tab_forecast, tab_data, tab_formulas = st.tabs(
    ["Overview", "Performance", "Risk", "Correlation", "Forecast", "Data", "Formulas"]
)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final value", f"${final_value:,.0f}")
    c2.metric("Total contributions", f"${total_contrib:,.0f}")
    c3.metric("Net profit", f"${net_profit:,.0f}")
    c4.metric("TWR CAGR", f"{twr_cagr:.2%}")

    c5, c6 = st.columns(2)
    c5.metric("TWR Total return", f"{twr_total_return:.2%}")
    c6.metric("Volatility (ann.)", f"{ann_vol:.2%}")

    fig_v = px.line(value, title="Portfolio value over time", labels={"value": "Value", "index": "Date"})
    st.plotly_chart(fig_v, use_container_width=True)

    alloc_plot = alloc.copy()
    alloc_plot.index.name = "Date"
    fig_alloc = px.area(alloc_plot, title="Allocation over time")
    fig_alloc.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_alloc, use_container_width=True)

with tab_performance:
    st.write("**(Coming next)** Monthly / Annual returns tables & charts.")

with tab_risk:
    st.subheader("Portfolio drawdown")
    fig_dd = px.area(
        dd_series,
        title=f"Drawdown (Max: {max_dd:.2%})",
        labels={"value": "Drawdown", "index": "Date"}
    )
    fig_dd.update_layout(yaxis_tickformat=".0%", yaxis_range=[-1, 0])
    st.plotly_chart(fig_dd, use_container_width=True)

with tab_corr:
    st.subheader("Correlation heatmap (assets)")
    corr = asset_rets.corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation of asset returns")
    st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------
# Forecast (portfolio-level Monte Carlo)
# -----------------------
with tab_forecast:
    st.subheader("Future Simulator (Monte Carlo — Portfolio Level)")

    colL, colR = st.columns(2)
    with colL:
        horizon_years = st.slider("Forecast horizon (years)", 1, 30, 10)
        n_sims = st.slider("# Simulations", 200, 5000, 2000, step=100)
    with colR:
        monthly_contribution = st.number_input(
            "Monthly contribution during forecast",
            min_value=0, value=int(st.session_state.get("forecast_monthly", 500)), step=100
        )
        st.session_state["forecast_monthly"] = monthly_contribution
        show_real = st.checkbox("Show in today's dollars (real)", value=False)
        infl_annual = st.number_input("Inflation (annual %)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)

    mu_d = float(twr.mean()) if len(twr) > 1 else 0.0
    sigma_d = float(twr.std(ddof=0)) if len(twr) > 1 else 0.0

    if sigma_d == 0 and mu_d == 0:
        st.info("Not enough variability in historical returns to simulate. Try a longer date range or different assets.")
    else:
        start_fc = value.index[-1] + pd.Timedelta(days=1)
        n_days = int(horizon_years * 252)
        fc_idx = pd.date_range(start=start_fc, periods=n_days, freq="B")

        s_tmp = pd.Series(1.0, index=fc_idx)
        month_firsts = s_tmp.groupby([fc_idx.year, fc_idx.month]).head(1).index
        contrib_vec = np.zeros(n_days, dtype=float)
        if monthly_contribution > 0:
            contrib_vec[[fc_idx.get_indexer([d])[0] for d in month_firsts]] = monthly_contribution

        pi = float(infl_annual) / 100.0
        t_days = np.arange(1, n_days + 1, dtype=float)
        deflator = (1.0 + pi) ** (t_days / 252.0)

        rng = np.random.default_rng()
        shocks = rng.normal(loc=mu_d, scale=sigma_d, size=(n_days, n_sims))

        V = np.empty((n_days, n_sims), dtype=float)
        V0 = float(value.iloc[-1])
        V_prev = np.full(n_sims, V0, dtype=float)

        for t_i in range(n_days):
            V_after_flow = V_prev + contrib_vec[t_i]
            V_t = V_after_flow * (1.0 + shocks[t_i, :])
            V[t_i, :] = V_t
            V_prev = V_t

        V_plot = V / deflator[:, None] if show_real else V

        pct = {
            "p05": np.percentile(V_plot, 5, axis=1),
            "p25": np.percentile(V_plot, 25, axis=1),
            "p50": np.percentile(V_plot, 50, axis=1),
            "p75": np.percentile(V_plot, 75, axis=1),
            "p95": np.percentile(V_plot, 95, axis=1),
        }
        fan_df = pd.DataFrame(pct, index=fc_idx)

        terminal = V_plot[-1, :]

        if show_real:
            contrib_pv = float((contrib_vec / deflator).sum())
            baseline = V0 + contrib_pv
        else:
            baseline = V0 + float(contrib_vec.sum())

        prob_loss_vs_baseline = float((terminal < baseline).mean())

        kpi_suffix = " (real)" if show_real else ""
        st.metric(f"Median terminal value{kpi_suffix}", f"${np.median(terminal):,.0f}")
        st.metric(f"5%–95% range{kpi_suffix}", f"${np.percentile(terminal,5):,.0f} — ${np.percentile(terminal,95):,.0f}")

        c1, c2 = st.columns(2)
        if show_real:
            c1.metric("PV of future contributions", f"${(contrib_vec / deflator).sum():,.0f}")
        else:
            c1.metric("Future contributions", f"${contrib_vec.sum():,.0f}")
        c2.metric(f"Prob. end < baseline{kpi_suffix}", f"{prob_loss_vs_baseline:.1%}")

        title_suffix = " (real)" if show_real else ""
        fan_plot = fan_df.rename(columns={"p50": "Median", "p25": "25th", "p75": "75th", "p05": "5th", "p95": "95th"})
        fig_fan = px.line(
            fan_plot[["Median", "25th", "75th", "5th", "95th"]],
            title=f"Projected Portfolio Value — Median & Percentile Bands{title_suffix}",
            labels={"value": "Value", "index": "Date"}
        )
        st.plotly_chart(fig_fan, use_container_width=True)

        hist_df = pd.DataFrame({"Terminal Value": terminal})
        fig_hist = px.histogram(hist_df, x="Terminal Value", nbins=50, title=f"Terminal Value Distribution{title_suffix}")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("### Projected Returns (Median Path)")
        median_series = fan_df["p50"]
        median_monthly = median_series.resample("M").last().pct_change().dropna()
        fig_mo = px.bar(median_monthly, title=f"Projected Median Monthly Returns{title_suffix}",
                        labels={"value": "Return", "index": "Month"})
        fig_mo.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig_mo, use_container_width=True)

        median_annual = median_series.resample("Y").last().pct_change().dropna()
        fig_yr = px.bar(median_annual, title=f"Projected Median Annual Returns{title_suffix}",
                        labels={"value": "Return", "index": "Year"})
        fig_yr.update_layout(yaxis_tickformat=".1%")
        st.plotly_chart(fig_yr, use_container_width=True)

with tab_data:
    st.write("**Prices (Adj Close)**")
    st.dataframe(prices.tail(), use_container_width=True)
    st.download_button(
        "Download prices CSV",
        data=prices.to_csv().encode("utf-8"),
        file_name="prices.csv",
        mime="text/csv"
    )

    st.write("**Portfolio value & flows**")
    out = pd.DataFrame({"value": value, "flow": flows})
    st.dataframe(out.tail(), use_container_width=True)
    st.download_button(
        "Download portfolio series CSV",
        data=out.to_csv().encode("utf-8"),
        file_name="portfolio_series.csv",
        mime="text/csv"
    )

with tab_formulas:
    st.subheader("Formulas")
    st.markdown(r"""
**Final value**  
Portfolio value on the last day:  
$$ \text{Final value} = V_{\text{end}} $$

---

**Total contributions**  
All money you put in (initial + monthly):  
$$ \text{Contributions} = \text{Initial} + \sum \text{Monthly contributions} $$

---

**Net profit**  
How much you gained beyond what you put in:  
$$ \text{Net profit} = \text{Final value} - \text{Contributions} $$

---

**Time-Weighted Daily Return (handles cash flows)**  
Return that removes the effect of deposits:  
$$ r_t = \frac{V_t - V_{t-1} - \text{flow}_t}{V_{t-1}} $$

---

**TWR Total return**  
Compounded growth from daily TWRs:  
$$ \text{TWR total return} = \prod_{t=1}^{T} (1 + r_t) - 1 $$

---

**TWR CAGR**  
Annualized growth rate over the whole period:  
$$ \text{TWR CAGR} = \left(\prod_{t=1}^{T} (1 + r_t)\right)^{1/\text{years}} - 1 $$

---

**Volatility (annualized)**  
Yearly “bumpiness” of returns:  
$$ \sigma_{\text{ann}} = \text{stdev}(r_t)\times\sqrt{252} $$

---

**Monte Carlo Forecast (Portfolio Level)**  
We estimate average daily return (μ) and volatility (σ) from history.  
Then we simulate many possible futures by “rolling the dice” each day:  
$$ V_t = (V_{t-1} + \text{contrib}_t) \times (1 + r_t), \quad r_t \sim N(\mu, \sigma) $$

---

**Inflation adjustment (today's dollars)**  
Daily deflator after \(t\) trading days:  
$$ \text{Deflator}_t = (1+\pi)^{t/252} $$
""")

# -----------------------
# Next steps
# -----------------------
st.caption("Next: add beta vs benchmark, Sharpe, IRR (money-weighted), and export of transactions.")
