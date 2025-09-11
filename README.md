# 📊 Stocks & FX Dashboard

A multipage **Streamlit web app** that provides a fast snapshot of global markets.  
Built with **Twelve Data** (stocks/ETFs) and **Frankfurter** (FX) APIs.

---

## 🚀 Features

- **KPI Overview** – quick performance metrics for major indices, ETFs, and FX pairs  
- **Watchlist** – customizable list of tickers with price history, 1D–3M lookbacks, and trend indicators  
- **Sparklines & Top Movers** – mini-charts for your tickers and top gainers/losers in your watchlist  
- **FX Snapshot** – latest exchange rates with 1-day % changes and historical sparklines  
- **Risk & Volatility** – rolling volatility, drawdowns, and risk metrics  
- **Monte Carlo Simulator** – estimate possible **future portfolio returns** using random simulations of price paths  

---

## 🎥 Demo

Click the image below to watch a short demo of the dashboard in action:

[![Watch the demo](demo/screenshot.png)](https://youtu.be/YOUR_VIDEO_ID)

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – fast interactive dashboards  
- [Twelve Data API](https://twelvedata.com/) – stock & ETF market data  
- [Frankfurter API](https://www.frankfurter.app/) – foreign exchange rates  
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) – data wrangling  
- [Plotly Express](https://plotly.com/python/plotly-express/) – interactive charts  

---

## 📂 Project Structure

Main components of the repository:

- **app.py** → Main entry page for Streamlit  
- **pages/** → Multipage Streamlit pages  
  - `1_Overview.py`  
  - `2_FX_Dashboard.py`  
  - `3_Risk_&_Volatility.py`  
  - `4_Monte_Carlo_Simulator.py`  
- **src/** → API clients & utilities  
  - `api/twelve_data.py`  
  - `api/frankfurter.py`  
- **models/** → Reusable charts and models  
- **requirements.txt** → Python dependencies  
- **.streamlit/** → Local config & secrets.toml (ignored in Git)  
- **README.md** → Project documentation  

---

## ⚡ Quickstart

1. **Clone the repo**  
   `git clone https://github.com/JonathanAvigdor/Stocks_dashboard.git && cd Stocks_dashboard`  

2. **Install dependencies**  
   `pip install -r requirements.txt`  

3. **Add your API keys**  
   Create a local `.streamlit/secrets.toml`:  
   ```toml
   TWELVEDATA_API_KEY = "YOUR_TWELVEDATA_KEY"

4. **⚠️ Never commit this file – it’s already in .gitignore.** 

Run the app
streamlit run app.py

## 🌐 Deployment

The app is deployed on Streamlit Cloud.
Every push to main redeploys automatically.

To deploy yourself:

1. Fork this repo

2. Go to Streamlit Cloud

3. Connect your GitHub repo → pick app.py as the main file

4. Set TWELVEDATA_API_KEY in Settings → Secrets

## 🔒 API Usage & Limits

* Twelve Data free tier: ~800 credits/day, 8/minute

* To avoid hitting limits:

  * Use the Low API mode toggle in the sidebar

  * Click Refresh manually instead of auto-refresh

  * Extend caching TTLs

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


