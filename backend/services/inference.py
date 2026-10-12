"""In-memory LightGBM inference for interactive forecast scenarios."""

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from backend.core.config import FEATURES_PATH, MODEL_PATH
from backend.schemas import SimulationRequest

CATEGORICAL_FEATURES = (
    "store",
    "item",
    "state_id",
    "dept_id",
    "cat_id",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
)


@lru_cache(maxsize=1)
def load_models():
    """Load the fitted LightGBM estimators once for the API process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {MODEL_PATH}. Run `python -m src.train`."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_feature_rows():
    """Load the precomputed baseline rows used for interactive what-if edits."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Feature data not found at {FEATURES_PATH}. Run `python -m src.features`."
        )
    return pd.read_csv(FEATURES_PATH, parse_dates=["date"])


def _feature_names(model):
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    booster = getattr(model, "booster_", None)
    if booster is not None:
        return booster.feature_name()
    raise ValueError("The loaded estimator does not expose its training feature names.")


def _category_map(model):
    """Map each categorical feature to the category order saved by LightGBM."""
    booster = getattr(model, "booster_", None)
    categories = getattr(booster, "pandas_categorical", None)
    feature_names = _feature_names(model)
    category_names = [name for name in feature_names if name in CATEGORICAL_FEATURES]
    return dict(zip(category_names, categories or []))


def _prepare_model_row(row, models):
    """Align the row and categorical dictionaries with the fitted estimator."""
    model = models["point"]
    feature_names = _feature_names(model)
    missing = [name for name in feature_names if name not in row.index]
    if missing:
        raise ValueError(f"Baseline feature row is missing model inputs: {missing}")
    # Construct columns individually: transposing a mixed pandas Series would
    # coerce every model input to object dtype, which LightGBM cannot predict on.
    aligned = pd.DataFrame({name: pd.Series([row[name]]) for name in feature_names})
    for name, categories in _category_map(model).items():
        aligned[name] = pd.Categorical(aligned[name], categories=categories)
    return aligned


def _set_event_override(row, value, models):
    """Set numeric and categorical calendar event fields consistently."""
    if "is_event" in row.index:
        row["is_event"] = int(value)
    if value:
        state = row.get("state_id")
        if pd.isna(state):
            state = str(row.get("store", "")).split("_")[0]
        for column in row.index:
            normalized = column.lower()
            if normalized.startswith("snap_"):
                row[column] = int(normalized == f"snap_{str(state).lower()}")
            elif normalized.startswith("is_holiday"):
                row[column] = 1
        for column in ("event_name_1", "event_type_1"):
            if column not in row.index:
                continue
            categories = _category_map(models["point"]).get(column)
            known = [
                category for category in categories or [] if str(category) != "None"
            ]
            if known:
                row[column] = known[0]


def _predict_quantiles(model_row, models):
    """Return nonnegative, noncrossing predictions for each available level."""
    levels = (0.10, 0.50, 0.75, 0.90, 0.95, 0.99)
    predicted = {
        level: max(0.0, float(models[f"q_{level}"].predict(model_row)[0]))
        for level in levels
        if f"q_{level}" in models
    }
    if 0.10 not in predicted or 0.90 not in predicted:
        raise ValueError("Model artifact must include P10 and P90 quantile models.")
    # Quantile tree models can cross after feature overrides. Project them back
    # to a monotone sequence so the displayed demand range stays coherent.
    running_max = 0.0
    for level in sorted(predicted):
        running_max = max(running_max, predicted[level])
        predicted[level] = running_max
    return predicted


def simulate_forecast(request: SimulationRequest, models=None, features=None):
    """Predict edited demand while keeping the trained input schema intact.

    This is real-time model inference over an existing baseline feature row. It
    does not create features for dates or products absent from features.csv.
    """
    models = load_models() if models is None else models
    features = load_feature_rows() if features is None else features
    target_date = pd.Timestamp(request.date)
    selected = features.loc[
        features["store"].eq(request.store_id)
        & features["item"].eq(request.item_id)
        & features["date"].eq(target_date)
    ]
    if selected.empty:
        raise LookupError(
            "No baseline feature row exists for this store, item, and date."
        )

    row = selected.iloc[0].copy()
    price_factor = 1.0 + request.price_change_pct / 100.0
    # sell_price is retained for auditing; the fitted model consumes its lag
    # and relative-price features, so mutate those model inputs as well.
    if "sell_price" in row.index and pd.notna(row["sell_price"]):
        row["sell_price"] = float(row["sell_price"]) * price_factor
    for column in row.index:
        if column.startswith("price_lag_") and pd.notna(row[column]):
            row[column] = float(row[column]) * price_factor
    if "price_ratio" in row.index and pd.notna(row["price_ratio"]):
        lag_columns = [name for name in row.index if name.startswith("price_lag_")]
        rolling_mean = row.get("price_rolling_mean_28")
        if lag_columns and pd.notna(rolling_mean):
            row["price_ratio"] = float(row[lag_columns[0]]) / (
                float(rolling_mean) + 1e-5
            )
        else:
            row["price_ratio"] = float(row["price_ratio"]) * price_factor
    _set_event_override(row, request.is_holiday_override, models)

    baseline_row = selected.iloc[0].copy()
    baseline_model_row = _prepare_model_row(baseline_row, models)
    model_row = _prepare_model_row(row, models)
    baseline_point = max(0.0, float(models["point"].predict(baseline_model_row)[0]))
    point = max(0.0, float(models["point"].predict(model_row)[0]))
    baseline_quantiles = _predict_quantiles(baseline_model_row, models)
    simulated_quantiles = _predict_quantiles(model_row, models)
    baseline_lower, baseline_upper = (
        baseline_quantiles[0.10],
        baseline_quantiles[0.90],
    )
    lower, upper = simulated_quantiles[0.10], simulated_quantiles[0.90]
    quantiles_changed = any(
        not np.isclose(
            baseline_quantiles[level], simulated_quantiles[level], rtol=1e-9, atol=1e-9
        )
        for level in baseline_quantiles
        if level in simulated_quantiles
    )
    return {
        "store_id": request.store_id,
        "item_id": request.item_id,
        "date": target_date.strftime("%Y-%m-%d"),
        "baseline_forecast": baseline_point,
        "baseline_p10": baseline_lower,
        "baseline_p90": baseline_upper,
        "simulated_forecast": point,
        "simulated_point": point,
        "simulated_p10": lower,
        "simulated_p90": upper,
        "baseline_quantiles": {
            f"p{int(level * 100)}": value for level, value in baseline_quantiles.items()
        },
        "simulated_quantiles": {
            f"p{int(level * 100)}": value
            for level, value in simulated_quantiles.items()
        },
        "prediction_changed": not np.isclose(
            baseline_point, point, rtol=1e-9, atol=1e-9
        ),
        "quantiles_changed": quantiles_changed,
        "price_change_pct": request.price_change_pct,
        "is_holiday_override": request.is_holiday_override,
    }
