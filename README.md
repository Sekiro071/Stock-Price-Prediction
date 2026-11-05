# Stock Price Prediction — NSE

A Flask web application that uses LSTM and GRU deep-learning models to forecast
the next trading day's **Open** and **Close** prices for any NSE-listed stock.

## Project Structure

```
Stock-Price-Prediction/
│
├── run.py                     # ← entry point  (python run.py)
│
├── app/
│   ├── __init__.py            # Flask app factory
│   ├── ml/
│   │   ├── data.py            # download, 70/15/15 split, MinMaxScaler, sequences
│   │   ├── models.py          # LSTM & GRU architectures, train / load, metrics
│   │   └── predictor.py       # top-level pipeline orchestrator
│   ├── routes/
│   │   └── predict.py         # Flask Blueprint  (GET / · POST /predict)
│   └── utils/
│       └── formatting.py      # format_large, safe_get, fmt_pct, clean_symbol
│
├── saved_models/              # auto-created; stores .keras + scaler.pkl per symbol
│
├── templates/
│   └── index.html             # single-page UI
│
├── static/                    # (css/ js/ — extend as needed)
│
└── requirements.txt
```

## Data Split Strategy

| Partition | Fraction | Purpose |
|-----------|----------|---------|
| **Train** | 70 %     | Fit `MinMaxScaler`; train LSTM & GRU weights |
| **Val**   | 15 %     | `EarlyStopping` monitor (`val_loss`, patience=5) |
| **Test**  | 15 %     | Untouched until final evaluation |

> [!IMPORTANT]
> The `MinMaxScaler` is **fit only on the training slice** and then applied to
> the entire dataset, preventing data leakage.

## Metrics Reported

RMSE and MAE are computed **separately for Open and Close** in actual **₹** values
on the held-out test set:

- `lstm_rmse_open` / `lstm_rmse_close`
- `lstm_mae_open`  / `lstm_mae_close`
- `gru_rmse_open`  / `gru_rmse_close`
- `gru_mae_open`   / `gru_mae_close`

The best model is selected by the lower **Close RMSE** (most user-visible metric).

## Model Persistence

Trained models are saved to `saved_models/<SYMBOL>_lstm.keras` and
`<SYMBOL>_gru.keras` on first run. Subsequent requests **reload from disk**
instead of retraining, keeping response time fast.

Models are automatically retrained if the number of data rows changes (new
trading days downloaded since last save).

## Quick Start

```bash
pip install -r requirements.txt
python run.py
# Open http://127.0.0.1:5000
```

