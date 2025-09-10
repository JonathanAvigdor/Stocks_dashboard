import requests, pandas as pd

BASE = "https://api.twelvedata.com"

class TwelveDataClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def daily(self, symbol: str, interval="1day", outputsize=100) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }
        r = requests.get(f"{BASE}/time_series", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if "values" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["values"])
        df.rename(columns={
            "datetime": "time",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        }, inplace=True)
        # Convert to correct types
        df["time"] = pd.to_datetime(df["time"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("time")
