"""
app/ml/predictor.py
-------------------
Top-level orchestration: download → split → scale → train/load → forecast.
Returns the complete result dict that the Flask route serialises to JSON.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.data import (
    download_data,
    split_and_scale,
    make_sequences,
    inverse_open_close,
    inverse_targets,
    apply_circuit_breaker,
)
from app.ml.models import train_or_load
from app.utils.formatting import safe_get, format_large, fmt_pct


def run_prediction(symbol: str) -> dict:
    """
    Full pipeline for one stock symbol.

    Data splits
    -----------
    train  70 %  — scaler fit + model training
    val    15 %  — EarlyStopping / model selection
    test   15 %  — untouched final evaluation

    Metrics returned (actual ₹)
    ---------------------------
    lstm_rmse_open, lstm_rmse_close, lstm_mae_open, lstm_mae_close
    gru_rmse_open,  gru_rmse_close,  gru_mae_open,  gru_mae_close
    """

    # ── 1. Download ──────────────────────────────────────────────────────────
    df, info = download_data(symbol)

    # ── 2. Split + Scale ─────────────────────────────────────────────────────
    scaled_data, raw_data, scaler, splits = split_and_scale(df)

    # ── 3. Build sequences ───────────────────────────────────────────────────
    X_train, y_train, X_val, y_val, X_test, y_test = make_sequences(
        scaled_data, splits
    )

    # ── 4. Train / Load + Evaluate ───────────────────────────────────────────
    result = train_or_load(
        symbol=symbol,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        scaler=scaler,
        df_len=len(df),
        inverse_fn=inverse_open_close,
    )

    best_model   = result["best_model"]
    best_name    = result["best_name"]
    best_real    = result["best_real"]     # (N_test, 5) ₹
    y_real       = result["y_real"]        # (N_test, 5) ₹
    lstm_metrics = result["lstm_metrics"]
    gru_metrics  = result["gru_metrics"]

    # ── 5. Next-day forecast ─────────────────────────────────────────────────
    last_60   = scaled_data[-60:].reshape(1, 60, 5)
    nxt_s     = best_model.predict(last_60, verbose=0)
    nxt_real  = inverse_open_close(nxt_s, scaler)

    current_close   = float(df["Close"].iloc[-1])
    raw_open        = float(nxt_real[0, 0])
    raw_close       = float(nxt_real[0, 3])
    predicted_open  = apply_circuit_breaker(raw_open,  current_close)
    predicted_close = apply_circuit_breaker(raw_close, current_close)
    clamped         = (predicted_open != raw_open) or (predicted_close != raw_close)
    change_pct      = (predicted_close - current_close) / current_close * 100
    direction       = "UP" if predicted_close >= current_close else "DOWN"

    # ── 6. Chart data ────────────────────────────────────────────────────────
    hist_dates = df.index.strftime("%Y-%m-%d").tolist()
    hist_close = [round(float(x), 2) for x in df["Close"].values.flatten()]

    # Test-set dates align with the last len(y_real) rows of df
    test_df    = df.iloc[splits["val_end"]:]
    test_dates = test_df.index[-len(y_real):].strftime("%Y-%m-%d").tolist()
    actual_close = [round(float(x), 2) for x in y_real[:, 3]]
    pred_close   = [round(float(x), 2) for x in best_real[:, 3]]

    # ── 7. Fundamentals ─────────────────────────────────────────────────────
    pe    = safe_get(info, "trailingPE", "forwardPE")
    pb    = safe_get(info, "priceToBook")
    mkcap = safe_get(info, "marketCap")
    eps   = safe_get(info, "trailingEps")
    div   = safe_get(info, "dividendYield")
    bv    = safe_get(info, "bookValue")
    promo = safe_get(info, "heldPercentInsiders")
    roe   = safe_get(info, "returnOnEquity")
    roce  = safe_get(info, "returnOnAssets")
    sector = safe_get(info, "sector", "industry", default="—")
    name   = safe_get(info, "longName", "shortName", default=symbol)

    high52    = safe_get(info, "fiftyTwoWeekHigh")
    low52     = safe_get(info, "fiftyTwoWeekLow")
    day_high  = safe_get(info, "dayHigh")
    day_low   = safe_get(info, "dayLow")
    prev_close = safe_get(info, "previousClose")

    day_change     = "N/A"
    day_change_pct = "N/A"
    if prev_close != "N/A":
        dc = current_close - float(prev_close)
        day_change     = f"{'+' if dc >= 0 else ''}{dc:.2f}"
        day_change_pct = f"{'+' if dc >= 0 else ''}{dc / float(prev_close) * 100:.2f}%"

    return {
        # ── Identity ────────────────────────────────────────────────────────
        "symbol":  symbol,
        "name":    name,
        "sector":  sector,

        # ── Current price ────────────────────────────────────────────────────
        "current_close":  round(current_close, 2),
        "day_change":     day_change,
        "day_change_pct": day_change_pct,
        "day_high":       round(float(day_high), 2) if day_high  != "N/A" else "N/A",
        "day_low":        round(float(day_low),  2) if day_low   != "N/A" else "N/A",
        "week52_high":    round(float(high52),   2) if high52    != "N/A" else "N/A",
        "week52_low":     round(float(low52),    2) if low52     != "N/A" else "N/A",

        # ── Prediction ───────────────────────────────────────────────────────
        "best_model":  best_name,
        "next_open":   round(predicted_open,  2),
        "next_close":  round(predicted_close, 2),
        "change_pct":  round(change_pct, 2),
        "direction":   direction,
        "clamped":     clamped,          # True when NSE circuit-breaker was applied

        # ── Per-model metrics (actual ₹) ─────────────────────────────────────
        "lstm_rmse_open":  round(lstm_metrics["rmse_open"],  2),
        "lstm_rmse_close": round(lstm_metrics["rmse_close"], 2),
        "lstm_mae_open":   round(lstm_metrics["mae_open"],   2),
        "lstm_mae_close":  round(lstm_metrics["mae_close"],  2),

        "gru_rmse_open":   round(gru_metrics["rmse_open"],   2),
        "gru_rmse_close":  round(gru_metrics["rmse_close"],  2),
        "gru_mae_open":    round(gru_metrics["mae_open"],    2),
        "gru_mae_close":   round(gru_metrics["mae_close"],   2),

        # ── Fundamentals ─────────────────────────────────────────────────────
        "pe":         round(float(pe),  2) if pe  != "N/A" else "N/A",
        "pb":         round(float(pb),  2) if pb  != "N/A" else "N/A",
        "eps":        round(float(eps), 2) if eps != "N/A" else "N/A",
        "mkcap":      format_large(mkcap),
        "div_yield":  fmt_pct(div)   if div   != "N/A" else "N/A",
        "book_value": round(float(bv), 2) if bv  != "N/A" else "N/A",
        "promo_hold": fmt_pct(promo) if promo != "N/A" else "N/A",
        "roe":        fmt_pct(roe)   if roe   != "N/A" else "N/A",
        "roce":       fmt_pct(roce)  if roce  != "N/A" else "N/A",

        # ── Chart ────────────────────────────────────────────────────────────
        "hist_dates":   hist_dates,
        "hist_close":   hist_close,
        "test_dates":   test_dates,
        "actual_close": actual_close,
        "pred_close":   pred_close,
    }

