import requests, pandas as pd

BASE = "https://www.alphavantage.co/query"

class AlphaVantageClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _get(self, params: dict) -> dict:
        # return raw JSON (no sleeps); caller will handle messages
        params = {**params, "apikey": self.api_key}
        r = requests.get(BASE, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def daily(self, symbol: str, outputsize: str = "compact") -> pd.DataFrame:
        data = self._get({
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize
        })

        # handle throttling / errors gracefully
        if "Note" in data:
            # rate limit hit
            return pd.DataFrame()
        if "Error Message" in data:
            # bad symbol, etc.
            return pd.DataFrame()

        ts = data.get("Time Series (Daily)", {})
        rows = []
        for dt, v in ts.items():
            rows.append({
                "time": dt,
                "open":  float(v["1. open"]),
                "high":  float(v["2. high"]),
                "low":   float(v["3. low"]),
                "close": float(v["4. close"]),
                "volume": float(v["5. volume"]),
            })
        return pd.DataFrame(rows)

    def quote(self, symbol: str) -> dict:
        data = self._get({"function": "GLOBAL_QUOTE", "symbol": symbol})
        return data.get("Global Quote", {})
