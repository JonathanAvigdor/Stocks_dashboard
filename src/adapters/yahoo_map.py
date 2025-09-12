# src/adapters/yahoo_map.py
from __future__ import annotations
from typing import List

# MIC -> Yahoo suffix
_MIC_SUFFIX = {
    "XSTO": ".ST",  # Nasdaq Stockholm
    "XTAE": ".TA",  # Tel Aviv
}

# Direct synonyms / special cases
_SPECIAL_PRIMARY = {
    # Stockholm index
    "OMXS30": "^OMXS30",
    "OMX30": "^OMXS30",
    "OMX": "^OMXS30",
    "^OMXS30": "^OMXS30",

    # XACT OMXS30 ETF (Stockholm)
    "XACT.OMXS30": "XACT-OMXS30.ST",
    "XACT OMXS30": "XACT-OMXS30.ST",
    "XACTOMXS30": "XACT-OMXS30.ST",
}

# Reasonable alternates to try if primary is empty
_SPECIAL_ALTERNATES = {
    "XACT.OMXS30": ["XACT-OMXS30.ST", "XACTOS"],
    "XACT OMXS30": ["XACT-OMXS30.ST", "XACTOS"],
    "XACTOMXS30":  ["XACT-OMXS30.ST", "XACTOS"],
    "OMXS30":      ["^OMXS30"],
    "OMX30":       ["^OMXS30"],
    "OMX":         ["^OMXS30"],
    "^OMXS30":     ["^OMXS30"],
}

def _norm(s: str) -> str:
    return (s or "").strip().upper()

def td_to_yahoo(sym: str) -> str:
    """
    Convert TD-style to Yahoo:
      - VOLV-B:XSTO -> VOLV-B.ST
      - TEVA:XTAE   -> TEVA.TA
      - OMXS30      -> ^OMXS30
      - XACT.OMXS30 -> XACT-OMXS30.ST
    """
    s = _norm(sym)
    if not s:
        return s
    if s in _SPECIAL_PRIMARY:
        return _SPECIAL_PRIMARY[s]
    if s.startswith("^"):
        return s
    s = s.replace(".", "-")
    if ":" in s:
        base, mic = s.split(":", 1)
        suf = _MIC_SUFFIX.get(mic.upper(), "")
        return base + suf if suf else base
    return s

def yahoo_alternates(original_sym: str, primary_yahoo: str) -> List[str]:
    s = _norm(original_sym)
    out: List[str] = []
    for a in _SPECIAL_ALTERNATES.get(s, []):
        if a != primary_yahoo and a not in out:
            out.append(a)
    return out
