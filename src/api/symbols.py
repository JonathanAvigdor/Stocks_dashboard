# src/api/symbols.py
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import time
import requests
import streamlit as st

TD_BASE = "https://api.twelvedata.com"

# ──────────────────────────────────────────────────────────────────────────────
# Exchanges & normalization
# ──────────────────────────────────────────────────────────────────────────────

# Sidebar options → MIC (None = US consolidated; plain SYMBOL works)
EXCHANGE_MAP = {
    "US (Consolidated)": None,
    "XSTO": "XSTO",
    "XTAE": "XTAE",
}

# Twelve Data sometimes returns an exchange **name** instead of a MIC.
# Add common synonyms so we can normalize reliably.
EXCHANGE_NAME_SYNONYMS: Dict[str, List[str]] = {
    "XSTO": [
        "Stockholm Stock Exchange",
        "Nasdaq Stockholm",
        "OMX Stockholm",
        "NASDAQ OMX Stockholm",
        "Stockholm",
    ],
    "XTAE": [
        "Tel Aviv Stock Exchange",
        "TASE",
        "Tel Aviv",
    ],
}

# Build quick reverse maps for name->MIC
NAME_TO_MIC: Dict[str, str] = {}
for mic, names in EXCHANGE_NAME_SYNONYMS.items():
    for n in names:
        NAME_TO_MIC[n] = mic

def _apikey() -> str:
    key = st.secrets.get("TWELVEDATA_API_KEY", "")
    if not key:
        raise RuntimeError("TWELVEDATA_API_KEY missing in Streamlit secrets.")
    return key

def _http_get(path: str, params: dict) -> dict:
    r = requests.get(f"{TD_BASE}/{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def _mic_of(exchange_str: str) -> str:
    """
    Normalize an exchange identifier to a MIC if recognized.
    Accepts MIC ('XSTO', 'XTAE') or common exchange names/synonyms ('TASE', 'Nasdaq Stockholm', ...).
    If unknown, returns input unchanged.
    """
    if not exchange_str:
        return exchange_str
    if exchange_str in EXCHANGE_MAP.values() or exchange_str in ("XSTO", "XTAE"):
        return exchange_str
    return NAME_TO_MIC.get(exchange_str, exchange_str)

# ──────────────────────────────────────────────────────────────────────────────
# Reference queries (cached)
# ──────────────────────────────────────────────────────────────────────────────

def _clean_row(it: dict) -> Dict:
    """Normalize fields from TD reference endpoints."""
    return {
        "symbol": (it.get("symbol") or "").strip(),
        "name": (it.get("name") or it.get("instrument_name") or "").strip(),
        "exchange": (it.get("exchange") or "").strip(),
        "country": (it.get("country") or "").strip(),
        "type": (it.get("type") or it.get("instrument_type") or "").strip(),
    }

@st.cache_data(ttl=86400, show_spinner=False)
def _td_stocks(prefix: str, exchange: Optional[str], country: Optional[str], limit: int) -> List[Dict]:
    """
    Primary: /stocks, try MIC; if empty and MIC has synonyms, retry each synonym.
    """
    params = {"symbol": prefix, "apikey": _apikey(), "format": "JSON"}
    rows: List[Dict] = []

    # Helper to fetch with optional exchange/country
    def fetch(ex_val: Optional[str]) -> List[Dict]:
        p = dict(params)
        if ex_val:
            p["exchange"] = ex_val
        if country:
            p["country"] = country
        data = _http_get("stocks", p)
        raw = data.get("data") if isinstance(data, dict) else data
        raw = raw or []
        return [_clean_row(it) for it in raw]

    # Try MIC first (or unfiltered if None)
    rows = fetch(exchange)
    if rows:
        return rows[:limit]

    # If filtered + empty, try exchange synonyms by name
    if exchange and exchange in EXCHANGE_NAME_SYNONYMS:
        for name in EXCHANGE_NAME_SYNONYMS[exchange]:
            rows = fetch(name)
            if rows:
                return rows[:limit]

    # As last resort, unfiltered stocks for the prefix
    if exchange:
        rows = fetch(None)
        if rows:
            return rows[:limit]

    return []

@st.cache_data(ttl=86400, show_spinner=False)
def _td_etf(prefix: str, exchange: Optional[str], country: Optional[str], limit: int) -> List[Dict]:
    """
    Optional: /etf reference (some tickers exist only as ETFs in TD).
    Same MIC/name fallback logic as _td_stocks.
    """
    params = {"symbol": prefix, "apikey": _apikey(), "format": "JSON"}
    rows: List[Dict] = []

    def fetch(ex_val: Optional[str]) -> List[Dict]:
        p = dict(params)
        if ex_val:
            p["exchange"] = ex_val
        if country:
            p["country"] = country
        data = _http_get("etf", p)
        raw = data.get("data") if isinstance(data, dict) else data
        raw = raw or []
        # Align to _clean_row
        out = []
        for it in raw:
            out.append({
                "symbol": (it.get("symbol") or "").strip(),
                "name": (it.get("name") or "").strip(),
                "exchange": (it.get("exchange") or "").strip(),
                "country": (it.get("country") or "").strip(),
                "type": "ETF",
            })
        return out

    rows = fetch(exchange)
    if rows:
        return rows[:limit]

    if exchange and exchange in EXCHANGE_NAME_SYNONYMS:
        for name in EXCHANGE_NAME_SYNONYMS[exchange]:
            rows = fetch(name)
            if rows:
                return rows[:limit]

    if exchange:
        rows = fetch(None)
        if rows:
            return rows[:limit]

    return []

@st.cache_data(ttl=86400, show_spinner=False)
def _td_symbol_search(prefix: str, limit: int) -> List[Dict]:
    """
    Broad fallback: /symbol_search (no exchange filter).
    We'll filter by exchange locally if needed.
    """
    data = _http_get("symbol_search", {
        "symbol": prefix,
        "apikey": _apikey(),
        "format": "JSON",
    })
    raw = data.get("data") if isinstance(data, dict) else data
    raw = raw or []
    out: List[Dict] = []
    for it in raw:
        out.append(_clean_row(it))
        if len(out) >= limit:
            break
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Minimal paid probe (1 bar) for stubborn symbols
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def _td_has_series(symbol: str) -> bool:
    """
    Validate by fetching 1 bar of time_series. Accept if 'values' non-empty.
    Tiny cost; only used when reference didn't confirm.
    """
    try:
        data = _http_get("time_series", {
            "symbol": symbol,
            "interval": "1day",
            "outputsize": 1,
            "apikey": _apikey(),
            "format": "JSON",
        })
        if isinstance(data, dict) and "values" in data:
            return bool(data.get("values"))
        return False
    except Exception:
        return False

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def format_suggestion_row(item: Dict) -> str:
    ex_raw = item.get("exchange") or ""
    ex_mic = _mic_of(ex_raw)
    country = item.get("country") or ""
    sym = item.get("symbol") or ""
    name = item.get("name") or ""
    left = f"{sym}" + (f":{ex_mic}" if ex_mic in ("XSTO", "XTAE") else "")
    suffix = f" ({country})" if country else ""
    return f"{left} — {name}{suffix}".strip()

def _combine_rank_limit(candidates: List[Dict], prefix: str, limit: int) -> List[Dict]:
    """
    Combine, de-duplicate by (symbol, normalized exchange), and lightly rank.
    Ranking:
      1) symbol startswith(prefix) > contains(prefix)
      2) type 'Common Stock' > 'ETF' > others
      3) shorter symbol first
    """
    seen = set()
    normed: List[Dict] = []
    prefix_u = prefix.upper()

    for it in candidates:
        sym = (it.get("symbol") or "").upper()
        ex = _mic_of(it.get("exchange") or "")
        key = (sym, ex)
        if key in seen or not sym:
            continue
        seen.add(key)
        it2 = dict(it)
        it2["exchange"] = ex
        normed.append(it2)

    def score(it: Dict) -> Tuple[int, int, int]:
        sym = (it.get("symbol") or "").upper()
        typ = (it.get("type") or "").lower()
        starts = 0 if sym.startswith(prefix_u) else 1
        # type order: stock (0), etf (1), other (2)
        if "stock" in typ:
            tscore = 0
        elif "etf" in typ:
            tscore = 1
        else:
            tscore = 2
        return (starts, tscore, len(sym))

    normed.sort(key=score)
    return normed[:limit]

def search_symbols(
    prefix: str,
    exchange: Optional[str] = None,
    country: Optional[str] = None,
    limit: int = 20,
) -> List[Dict]:
    """
    Suggestions with robust fallbacks and ETF coverage:
      1) /stocks (MIC → synonyms) + /etf combined
      2) If empty, unfiltered /stocks + /etf for the prefix
      3) If still empty, /symbol_search (filter by exchange MIC/name if provided)
      4) If filtered empty, show global symbol_search results
    Results are de-duplicated, ranked, and capped to `limit`.
    """
    prefix = (prefix or "").strip()
    if not prefix:
        return []

    mic = exchange
    # Try filtered stocks + etf (MIC and names handled inside helpers)
    stock_rows = _td_stocks(prefix=prefix, exchange=mic, country=country, limit=limit)
    etf_rows   = _td_etf(prefix=prefix, exchange=mic, country=country, limit=limit)
    combined = stock_rows + etf_rows
    if combined:
        return _combine_rank_limit(combined, prefix, limit)

    # Unfiltered stocks/etf by prefix (as rescue)
    if exchange:
        stock_rows = _td_stocks(prefix=prefix, exchange=None, country=country, limit=limit)
        etf_rows   = _td_etf(prefix=prefix, exchange=None, country=country, limit=limit)
        combined = stock_rows + etf_rows
        if combined:
            return _combine_rank_limit(combined, prefix, limit)

    # symbol_search broad fallback
    ss = _td_symbol_search(prefix=prefix, limit=50)
    if exchange:
        # Accept either MIC or any of its synonyms
        names = EXCHANGE_NAME_SYNONYMS.get(exchange, [])
        filtered = [r for r in ss if (r.get("exchange") in ([exchange] + names))]
        if filtered:
            return _combine_rank_limit(filtered, prefix, limit)
        # nothing matched → show global results so user still sees likely hits
        return _combine_rank_limit(ss, prefix, limit)

    return _combine_rank_limit(ss, prefix, limit)

def normalize_symbol(symbol: str, exchange_hint: Optional[str] = None) -> str:
    """
    Normalize to TD format:
      - US symbols: plain 'AAPL'
      - Non-US commonly require 'SYMBOL:MIC' (e.g., 'VOLV-B:XSTO', 'TEVA:XTAE')
    If symbol already contains ':', return as-is.
    """
    s = (symbol or "").strip().upper()
    if not s:
        return s
    if ":" in s:
        return s
    mic = _mic_of(exchange_hint or "")
    if mic in ("XSTO", "XTAE"):
        return f"{s}:{mic}"
    return s

def _reference_hit(symbol_base: str, exchange: Optional[str]) -> bool:
    """
    Use the robust search (cached) to confirm presence. Accept either MIC or known name.
    """
    rows = search_symbols(prefix=symbol_base, exchange=exchange, country=None, limit=50)
    if not rows:
        return False
    if exchange in ("XSTO", "XTAE"):
        accepted = set([exchange] + EXCHANGE_NAME_SYNONYMS.get(exchange, []))
        return any(
            r.get("symbol", "").upper() == symbol_base and (r.get("exchange") in accepted)
            for r in rows
        )
    return any(r.get("symbol", "").upper() == symbol_base for r in rows)

def validate_symbol_supported(normalized_symbol: str) -> Tuple[bool, Optional[str]]:
    """
    1) Reference check (cached)
    2) If that fails, tiny 1-bar time_series probe for the exact normalized symbol
    """
    if not normalized_symbol:
        return False, "Empty symbol."
    if ":" in normalized_symbol:
        base, venue = normalized_symbol.split(":", 1)
        if _reference_hit(base, venue):
            return True, None
    else:
        if _reference_hit(normalized_symbol, None):
            return True, None

    # Fallback: paid but minimal probe
    if _td_has_series(normalized_symbol):
        return True, None

    return False, "Symbol not found in reference data."

def ensure_session_state():
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []
    if "_last_query_ts" not in st.session_state:
        st.session_state._last_query_ts = 0.0

def add_to_watchlist(symbol: str, exchange_hint: Optional[str] = None) -> Tuple[bool, Optional[str], str]:
    """
    Normalize, validate, deduplicate. Updates session_state.watchlist on success.
    Returns (ok, error_message, normalized_symbol)
    """
    ensure_session_state()
    norm = normalize_symbol(symbol, exchange_hint)
    ok, err = validate_symbol_supported(norm)
    if not ok:
        return False, err, ""
    if norm in st.session_state.watchlist:
        return False, "Already in watchlist.", norm
    st.session_state.watchlist.append(norm)
    return True, None, norm

def debounce_ok(ms: int = 400) -> bool:
    """
    Simple debounce gate to limit reference calls during typing.
    """
    now = time.time()
    if now - st.session_state.get("_last_query_ts", 0.0) < (ms / 1000.0):
        return False
    st.session_state._last_query_ts = now
    return True
