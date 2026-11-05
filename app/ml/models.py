"""
app/ml/models.py
----------------
Build, train, and evaluate LSTM / GRU models.

Key design decisions
--------------------
* EarlyStopping monitors *val_loss* (true held-out validation set).
* Models are saved to `saved_models/<SYMBOL>_lstm.keras` and `..._gru.keras`
  so Flask reloads them on subsequent requests instead of retraining.
* RMSE and MAE are reported **separately** for Open and Close in actual ₹.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Lazy TensorFlow imports — avoids startup cost on non-prediction code paths
def _tf_imports():
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    return tf, Sequential, LSTM, GRU, Dense, Dropout, EarlyStopping


SAVED_MODELS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "saved_models"
)


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _model_paths(symbol: str) -> dict[str, str]:
    slug = symbol.replace(".", "_").upper()
    base = os.path.abspath(SAVED_MODELS_DIR)
    return {
        "lstm":   os.path.join(base, f"{slug}_lstm.keras"),
        "gru":    os.path.join(base, f"{slug}_gru.keras"),
        "scaler": os.path.join(base, f"{slug}_scaler.pkl"),
        "meta":   os.path.join(base, f"{slug}_meta.pkl"),
    }


def _models_are_fresh(paths: dict[str, str], df_len: int) -> bool:
    """
    Return True only when all four artefacts exist AND the saved data-length
    matches the current download (guards against stale models after new data).
    """
    if not all(os.path.exists(p) for p in paths.values()):
        return False
    try:
        with open(paths["meta"], "rb") as f:
            meta = pickle.load(f)
        return meta.get("df_len") == df_len
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Architecture factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_lstm(input_shape: tuple) -> "Sequential":
    _, Sequential, LSTM, GRU, Dense, Dropout, _ = _tf_imports()
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.1),
        LSTM(32),
        Dropout(0.1),
        Dense(2),
    ], name="LSTM_model")
    model.compile(optimizer="adam", loss="mse")
    return model


def _build_gru(input_shape: tuple) -> "Sequential":
    _, Sequential, LSTM, GRU, Dense, Dropout, _ = _tf_imports()
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.1),
        GRU(32),
        Dropout(0.1),
        Dense(2),
    ], name="GRU_model")
    model.compile(optimizer="adam", loss="mse")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true_real: np.ndarray,   # (N, 5) — inverse-transformed ground truth
    y_pred_real: np.ndarray,   # (N, 5) — inverse-transformed predictions
) -> dict:
    """
    Compute RMSE and MAE separately for Open (col 0) and Close (col 3)
    in actual ₹ values.
    """
    open_true,  close_true  = y_true_real[:, 0], y_true_real[:, 3]
    open_pred,  close_pred  = y_pred_real[:, 0], y_pred_real[:, 3]

    return {
        "rmse_open":  float(np.sqrt(mean_squared_error(open_true,  open_pred))),
        "rmse_close": float(np.sqrt(mean_squared_error(close_true, close_pred))),
        "mae_open":   float(mean_absolute_error(open_true,  open_pred)),
        "mae_close":  float(mean_absolute_error(close_true, close_pred)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def train_or_load(
    symbol: str,
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    scaler,
    df_len: int,
    inverse_fn,          # callable: (pred_scaled, scaler) → real array (N,5)
) -> dict:
    """
    Train LSTM and GRU (or load from disk if artefacts are fresh).
    Runs final evaluation on the untouched test set.

    Returns a result dict consumed by the prediction route.
    """
    from tensorflow.keras.models import load_model
    tf, Sequential, LSTM, GRU, Dense, Dropout, EarlyStopping = _tf_imports()

    paths = _model_paths(symbol)

    # ── Load or Train ────────────────────────────────────────────────────────
    if _models_are_fresh(paths, df_len):
        lstm_model = load_model(paths["lstm"])
        gru_model  = load_model(paths["gru"])
    else:
        input_shape = (X_train.shape[1], X_train.shape[2])  # (60, 5)

        es = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        lstm_model = _build_lstm(input_shape)
        lstm_model.fit(
            X_train, y_train,
            epochs=50, batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[es], verbose=0,
        )

        gru_model = _build_gru(input_shape)
        gru_model.fit(
            X_train, y_train,
            epochs=50, batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[es], verbose=0,
        )

        # Persist artefacts
        os.makedirs(os.path.abspath(SAVED_MODELS_DIR), exist_ok=True)
        lstm_model.save(paths["lstm"])
        gru_model.save(paths["gru"])
        with open(paths["scaler"], "wb") as f:
            pickle.dump(scaler, f)
        with open(paths["meta"], "wb") as f:
            pickle.dump({"df_len": df_len}, f)

    # ── Evaluate on untouched test set ───────────────────────────────────────
    lstm_pred_s = lstm_model.predict(X_test, verbose=0)   # (N_test, 2) scaled
    gru_pred_s  = gru_model.predict(X_test,  verbose=0)

    lstm_real = inverse_fn(lstm_pred_s, scaler)            # (N_test, 5) ₹
    gru_real  = inverse_fn(gru_pred_s,  scaler)
    y_real    = inverse_fn(y_test,       scaler)           # ground truth ₹

    lstm_metrics = compute_metrics(y_real, lstm_real)
    gru_metrics  = compute_metrics(y_real, gru_real)

    # Best model = lower Close RMSE (most user-visible metric)
    if lstm_metrics["rmse_close"] <= gru_metrics["rmse_close"]:
        best_name   = "LSTM"
        best_model  = lstm_model
        best_pred_s = lstm_pred_s
        best_real   = lstm_real
    else:
        best_name   = "GRU"
        best_model  = gru_model
        best_pred_s = gru_pred_s
        best_real   = gru_real

    return {
        "lstm_model":    lstm_model,
        "gru_model":     gru_model,
        "best_name":     best_name,
        "best_model":    best_model,
        "best_pred_s":   best_pred_s,
        "best_real":     best_real,
        "y_real":        y_real,
        "lstm_metrics":  lstm_metrics,
        "gru_metrics":   gru_metrics,
    }

