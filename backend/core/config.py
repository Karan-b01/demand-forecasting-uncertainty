"""Application settings and paths derived from the repository root."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = PROJECT_ROOT / "data/processed/predictions_calibrated.csv"
FEATURES_PATH = PROJECT_ROOT / "data/processed/features.csv"
MODEL_PATH = PROJECT_ROOT / "models/lgbm_models.joblib"

API_TITLE = "Demand Forecasting API"
API_VERSION = "1.0.0"
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
