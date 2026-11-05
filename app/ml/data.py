"""
app/ml/data.py
--------------
Data download, chronological splitting, and sequence building.

Split strategy
--------------
  train  : 70 %  — scaler is fit exclusively on this portion
  val    : 15 %  — used for EarlyStopping / model selection
  test   : 15 %  — untouched until final evaluation

Sequence look-back: 60 trading days.
Features: Open, High, Low, Close, Volume  (index order 0-4).
Targets : Open (col 0), Close (col 3).

Caching
-------
Downloaded DataFrames are cached in memory keyed by (symbol, trading-date).
This means a second request for the same symbol on the same trading day skips
the yfinance network round-trip (~1-2 s saved).
"""

from __future__ import annotations

import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

LOOK_BACK  = 60
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 1 - TRAIN_FRAC - VAL_FRAC  = 0.15  (implicit)

FEATURES   = ["Open", "High", "Low", "Close", "Volume"]
N_FEATURES = len(FEATURES)   # 5

# NSE circuit-breaker cap: price cannot move more than 20 % in a single day.
NSE_CIRCUIT_PCT = 0.20


# ─────────────────────────────────────────────────────────────────────────────
# Download — with same-day in-process cache
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _cached_download(symbol: str, trading_date: str):
    """
    Inner cached download.  `trading_date` is today's date string — it makes
    the cache key stale automatically the next trading day, so stale data is
    never served beyond one session.
    """
    ticker = yf.Ticker(symbol)
    df = yf.download(
        symbol,
        start="2021-01-01",
        end=pd.Timestamp.today().strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )
    info = ticker.info or {}
    return df, info


def download_data(symbol: str) -> tuple[pd.DataFrame, dict]:
    """
    Download OHLCV history and ticker info from Yahoo Finance.
    Results are cached in-process for the current trading day.
    """
    today = datetime.date.today().isoformat()
    df, info = _cached_download(symbol, today)

    if df.empty or len(df) < 120:
        raise ValueError(
            f"Not enough data for {symbol}. "
            "Need at least 120 trading days. Try a different stock."
        )

    # Flatten MultiIndex columns produced by some yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.copy(), dict(info)


# ─────────────────────────────────────────────────────────────────────────────
# Chronological split + scaling
# ─────────────────────────────────────────────────────────────────────────────

def split_and_scale(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, dict[str, int]]:
    """
    Returns
    -------
    scaled_data : ndarray  shape (N, 5)   — full scaled array (train|val|test)
    raw_data    : ndarray  shape (N, 5)   — full raw array (for inverse reference)
    scaler      : MinMaxScaler fit only on training rows
    splits      : dict with boundary indices {train_end, val_end}
                  train = [:train_end]
                  val   = [train_end:val_end]
                  test  = [val_end:]
    """
    raw_data = df[FEATURES].values.astype(float)  # (N, 5)
    N = len(raw_data)

    train_end = int(N * TRAIN_FRAC)
    val_end   = int(N * (TRAIN_FRAC + VAL_FRAC))

    # Fit scaler ONLY on train slice
    scaler = MinMaxScaler()
    scaler.fit(raw_data[:train_end])

    scaled_data = scaler.transform(raw_data)       # transform all rows

    splits = {"train_end": train_end, "val_end": val_end}
    return scaled_data, raw_data, scaler, splits


# ─────────────────────────────────────────────────────────────────────────────
# Sequence builder
# ─────────────────────────────────────────────────────────────────────────────

def make_sequences(
    scaled_data: np.ndarray,
    splits: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build overlapping look-back windows; return train / val / test splits.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    Each X : (samples, LOOK_BACK, N_FEATURES)
    Each y : (samples, 2)   → [Open_scaled, Close_scaled]
    """
    X_all, y_all = [], []
    for i in range(LOOK_BACK, len(scaled_data)):
        X_all.append(scaled_data[i - LOOK_BACK : i])                         # (60, 5)
        y_all.append([scaled_data[i][0], scaled_data[i][3]])                 # Open, Close

    X_all = np.array(X_all)   # (N - LOOK_BACK, 60, 5)
    y_all = np.array(y_all)   # (N - LOOK_BACK, 2)

    # Sequence index 0 corresponds to raw row LOOK_BACK.
    # Adjust boundaries: a sequence at position i predicts raw row i + LOOK_BACK.
    train_end_seq = splits["train_end"] - LOOK_BACK   # sequences whose target is in train
    val_end_seq   = splits["val_end"]   - LOOK_BACK

    X_train, y_train = X_all[:train_end_seq],              y_all[:train_end_seq]
    X_val,   y_val   = X_all[train_end_seq:val_end_seq],   y_all[train_end_seq:val_end_seq]
    X_test,  y_test  = X_all[val_end_seq:],                y_all[val_end_seq:]

    return X_train, y_train, X_val, y_val, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Inverse-transform helpers
# ─────────────────────────────────────────────────────────────────────────────

def inverse_open_close(
    pred_scaled: np.ndarray,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """
    Invert scaled [Open, Close] predictions back to ₹ values.

    pred_scaled : (N, 2)  — columns are [Open_scaled, Close_scaled]
    Returns     : (N, 5)  — full reconstructed array; cols 0 (Open) and 3 (Close)
                            contain the actual rupee values; others are zeros.
    """
    dummy = np.zeros((len(pred_scaled), N_FEATURES))
    dummy[:, 0] = pred_scaled[:, 0]   # Open
    dummy[:, 3] = pred_scaled[:, 1]   # Close
    return scaler.inverse_transform(dummy)


def inverse_targets(
    y_scaled: np.ndarray,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """Invert ground-truth scaled targets; identical layout to inverse_open_close."""
    return inverse_open_close(y_scaled, scaler)


# ─────────────────────────────────────────────────────────────────────────────
# NSE circuit-breaker clamp
# ─────────────────────────────────────────────────────────────────────────────

def apply_circuit_breaker(
    predicted: float,
    reference: float,
    max_pct: float = NSE_CIRCUIT_PCT,
) -> float:
    """
    Clamp `predicted` price so it cannot deviate more than `max_pct` from
    `reference` (previous close).

    NSE enforces ±20 % intraday circuit limits on individual stocks, so any
    model prediction outside this band is physically impossible during regular
    trading.  We clamp rather than reject so the UI always gets a usable value.
    """
    lower = reference * (1.0 - max_pct)
    upper = reference * (1.0 + max_pct)
    return float(np.clip(predicted, lower, upper))
