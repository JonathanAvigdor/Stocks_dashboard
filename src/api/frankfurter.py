import requests, pandas as pd

BASE = "https://api.frankfurter.app"

class FrankfurterClient:
    def latest(self, base: str = "USD", symbols: str = "EUR"):
        r = requests.get(f"{BASE}/latest", params={"base": base, "symbols": symbols}, timeout=30)
        r.raise_for_status()
        return r.json()

    def timeseries(self, start: str, end: str, base: str = "USD", symbols: str = "EUR") -> pd.DataFrame:
        r = requests.get(f"{BASE}/{start}..{end}", params={"base": base, "symbols": symbols}, timeout=30)
        r.raise_for_status()
        data = r.json()
        rows = [{"time": dt, "rate": list(rate_map.values())[0]} for dt, rate_map in data.get("rates", {}).items()]
        return pd.DataFrame(rows).sort_values("time")

