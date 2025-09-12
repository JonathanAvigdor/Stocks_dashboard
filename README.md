# 📊 Stocks & FX Dashboard 

A multipage Streamlit web app that gives a fast snapshot of global markets.
Prices & fundamentals: yfinance (Yahoo Finance) • FX: Frankfurter API.
No paid market-data key required. (Optional: Twelve Data only for smarter symbol search.)

## 🔗 Live App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jonathanavigdor-stocks-dashboard-app-zzz14s.streamlit.app/Portfolio_Simulator)

- **Direct link to Portfolio Simulator:**  
  https://jonathanavigdor-stocks-dashboard-app-zzz14s.streamlit.app/Portfolio_Simulator

---

## 🚀 Features

- KPI Overview – quick % change vs last close for major ETFs/indices + headline FX
- Global Watchlist – add US & non-US tickers; see last price and 1D/5D/21D/63D lookbacks
- Sparklines & Top Movers – mini trend charts + top gainers/losers in your watchlist
- FX Snapshot – latest rates + tiny historical sparkline (Frankfurter)
- Risk & Volatility – rolling vol, drawdowns, and correlation heatmap
- Portfolio Simulator – backtest with contributions & rebalancing + Monte Carlo future paths
- Global Symbol Search – add equities/ETFs from US, Sweden (XSTO), Israel (XTAE), etc.
- You can type either Yahoo or “SYMBOL:MIC” — the app normalizes/auto-maps.

---

## 🎥 Demo

Click the image below to watch a short demo of the dashboard in action:

[![Watch the demo](demo/screenshot.png)](https://youtu.be/X6iAGP6US4E)

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – fast interactive dashboards  
- [Twelve Data API](https://twelvedata.com/) – stock & ETF market data  
- [Frankfurter API](https://www.frankfurter.app/) – foreign exchange rates  
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) – data wrangling  
- [Plotly Express](https://plotly.com/python/plotly-express/) – interactive charts  

---

Stocks_dashboard (project root)
├─ app.py
├─ pages
│  ├─ 1_Overview.py
│  ├─ 2_Stock_Explorer.py
│  ├─ 3_Risk_&_Volatility.py
│  └─ 4_Portfolio_Simulator.py
├─ models
│  └─ functions.py
├─ src
│  ├─ adapters
│  │  └─ yahoo_map.py
│  └─ api
│     ├─ frankfurter.py
│     └─ symbols.py
├─ demo
│  └─ screenshot.png
├─ .streamlit
│  ├─ config.toml
│  └─ secrets.toml   (do not commit)
├─ requirements.txt
└─ README.md



---
## ⚡ Quickstart

1. **Clone the repo**  
    git clone https://github.com/<your-username>/<your-repo>.git
    cd <your-repo>

2. **Install dependencies**  
   `pip install -r requirements.txt`  

3. **run**
   streamlit run app.py
   
## 🌍 Symbols & Examples

The app auto-normalizes common formats to Yahoo tickers:
- US: AAPL, MSFT, SPY
- Sweden (XSTO): type VOLV-B:XSTO or Yahoo VOLV-B.ST
- OMXS30 index: type OMXS30 (auto-mapped to ^OMXS30)
- OMXS30 ETF: XACT.OMXS30 → XACT-OMXS30.ST (auto-mapped)

If a symbol fails once, try the Yahoo form directly.


## 📈 Monte Carlo Simulator

The Monte Carlo simulator page models the uncertain future returns of a stock or portfolio:

* Estimates volatility and returns from historical data

* Runs thousands of random price path simulations

* Produces confidence intervals (5%, 50%, 95%) and visualizations

* Helps investors understand the range of risks and rewards instead of relying on a single forecast

## 👤 Author

Jonathan Avigdor

📍 Sweden 

💼 Civil Engineer & Data Enthusiast

🌐 [LinkedIn](https://www.linkedin.com/in/jonathanavigdor/)

## 📜 License

MIT License – free to use and modify.


