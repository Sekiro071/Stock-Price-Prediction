"""
app/routes/predict.py
---------------------
Flask Blueprint that exposes:
  GET  /                 → renders index.html
  GET  /api/companies    → returns full NSE company list as JSON (for autocomplete)
  POST /predict          → runs prediction pipeline, returns JSON
"""

import warnings
warnings.filterwarnings("ignore")

import json
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Blueprint, jsonify, render_template, request, current_app

from app.ml.predictor import run_prediction
from app.utils.formatting import clean_symbol

predict_bp = Blueprint("predict", __name__)

# Path to the static company list
_COMPANIES_JSON = os.path.join(
    os.path.dirname(__file__), "..", "..", "static", "data", "nse_companies.json"
)
# Cache in module-level variable — loaded once at first request
_companies_cache: list | None = None


def _load_companies() -> list:
    global _companies_cache
    if _companies_cache is None:
        with open(_COMPANIES_JSON, encoding="utf-8") as f:
            _companies_cache = json.load(f)
    return _companies_cache


@predict_bp.route("/")
def index():
    return render_template("index.html")


@predict_bp.route("/api/companies")
def companies():
    """Return the full NSE company list for autocomplete."""
    return jsonify(_load_companies())


@predict_bp.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    raw  = body.get("symbol", "").strip()

    if not raw:
        return jsonify({"error": "Please enter a stock symbol."}), 400

    symbol = clean_symbol(raw)
    try:
        result = run_prediction(symbol)
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

