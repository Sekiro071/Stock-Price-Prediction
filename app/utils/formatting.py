"""
app/utils/formatting.py
-----------------------
Pure formatting helpers — no ML, no Flask imports.
"""

import numpy as np


def safe_get(d: dict, *keys, default="N/A"):
    """Return the first non-null, non-empty, finite value from *keys* in dict *d*."""
    for k in keys:
        v = d.get(k)
        if v is not None and v != "":
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                continue
            return v
    return default


def format_large(num, default="N/A") -> str:
    """Format large INR values with human-readable suffixes."""
    if num in ("N/A", None):
        return "N/A"
    try:
        num = float(num)
        if np.isnan(num):
            return "N/A"
        if num >= 1e12:
            return f"₹{num / 1e12:.2f}T Cr."
        if num >= 1e7:
            return f"₹{num / 1e7:.2f} L Cr."
        if num >= 1e5:
            return f"₹{num / 1e5:.2f} L"
        return f"₹{num:,.2f}"
    except Exception:
        return "N/A"


def fmt_pct(v) -> str:
    """Format a decimal ratio as a percentage string."""
    if v == "N/A":
        return "N/A"
    v = float(v)
    return f"{v * 100:.2f}%" if abs(v) < 10 else f"{v:.2f}%"


def clean_symbol(raw: str) -> str:
    """Normalise a user-supplied symbol to <TICKER>.NS format."""
    raw = raw.strip().upper()
    if not raw.endswith(".NS"):
        raw += ".NS"
    return raw

