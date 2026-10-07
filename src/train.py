import os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def prepare_data_splits(features_csv: str = "data/processed/features.csv"):
  """Loads features and splits strictly by time into Train, Validation, and Test."""
  print(f"Loading feature dataset from {features_csv}...")
  df = pd.read_csv(features_csv)
  df["date"] = pd.to_datetime(df["date"])
  df = df.sort_values(["date", "store", "item"]).reset_index(drop=True)

  # Target and feature column definitions
  target_col = "sales"
  ignore_cols = [
      "date",
      "id",
      "d",
      "state_id",
      "sales",
      "store",
      "item",
      "dept_id",
      "cat_id",
  ]
  feature_cols = [c for c in df.columns if c not in ignore_cols]

  # Ensure categorical columns are properly typed for LightGBM
  for col in ["event_name_1", "event_type_1"]:
    if col in feature_cols:
      df[col] = df[col].astype("category")

  # Split dates: Final 28 days = Test, prior 28 days = Validation
  max_date = df["date"].max()
  test_start = max_date - pd.Timedelta(days=27)
  val_start = test_start - pd.Timedelta(days=28)

  train_mask = df["date"] < val_start
  val_mask = (df["date"] >= val_start) & (df["date"] < test_start)
  test_mask = df["date"] >= test_start

  print(
      f"Train span:      {df.loc[train_mask, 'date'].min().date()} to"
      f" {df.loc[train_mask, 'date'].max().date()} ({train_mask.sum():,} rows)"
  )
  print(
      f"Validation span: {df.loc[val_mask, 'date'].min().date()} to"
      f" {df.loc[val_mask, 'date'].max().date()} ({val_mask.sum():,} rows)"
  )
  print(
      f"Test span:       {df.loc[test_mask, 'date'].min().date()} to"
      f" {df.loc[test_mask, 'date'].max().date()} ({test_mask.sum():,} rows)"
  )

  splits = {
      "X_train": df.loc[train_mask, feature_cols],
      "y_train": df.loc[train_mask, target_col],
      "X_val": df.loc[val_mask, feature_cols],
      "y_val": df.loc[val_mask, target_col],
      "X_test": df.loc[test_mask, feature_cols],
      "y_test": df.loc[test_mask, target_col],
      "meta_val": df.loc[val_mask, ["date", "store", "item", "sales"]],
      "meta_test": df.loc[test_mask, ["date", "store", "item", "sales"]],
      "feature_cols": feature_cols,
  }
  return splits


def train_models(splits: dict):
  """Trains Point model (RMSE) and Quantile models (0.1, 0.5, 0.9)."""
  X_train, y_train = splits["X_train"], splits["y_train"]
  X_val, y_val = splits["X_val"], splits["y_val"]
  X_test, y_test = splits["X_test"], splits["y_test"]

  models = {}
  val_preds = {}
  test_preds = {}

  # 1. Point forecast model (RMSE)
  print("\n--- Training Point Model (L2/RMSE) ---")
  point_model = lgb.LGBMRegressor(
      objective="regression",
      metric="rmse",
      learning_rate=0.05,
      n_estimators=400,
      random_state=42,
      verbose=-1,
  )
  point_model.fit(
      X_train,
      y_train,
      eval_set=[(X_val, y_val)],
      callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
  )
  models["point"] = point_model
  val_preds["point"] = point_model.predict(X_val)
  test_preds["point"] = point_model.predict(X_test)

  # 2. Quantile models: 0.1, 0.5, 0.9
  quantiles = [0.1, 0.5, 0.9]
  for q in quantiles:
    print(f"--- Training Quantile Model (alpha={q}) ---")
    q_model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=q,
        metric="quantile",
        learning_rate=0.05,
        n_estimators=400,
        random_state=42,
        verbose=-1,
    )
    q_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    models[f"q_{q}"] = q_model
    val_preds[f"q_{q}"] = q_model.predict(X_val)
    test_preds[f"q_{q}"] = q_model.predict(X_test)

  return models, val_preds, test_preds


def enforce_monotonicity(
    preds_dict: dict, prefix: str = ""
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Enforces non-crossing constraint: q10 <= q50 <= q90 and non-negativity."""
  q10 = np.maximum(0.0, preds_dict["q_0.1"])
  q50 = np.maximum(0.0, preds_dict["q_0.5"])
  q90 = np.maximum(0.0, preds_dict["q_0.9"])

  # Sort across quantiles per row to eliminate crossing
  stacked = np.sort(np.vstack([q10, q50, q90]), axis=0)
  return stacked[0, :], stacked[1, :], stacked[2, :]


def run_pipeline():
  os.makedirs("models", exist_ok=True)
  os.makedirs("data/processed", exist_ok=True)

  splits = prepare_data_splits()
  models, val_preds, test_preds = train_models(splits)

  # Save trained models
  joblib.dump(models, "models/lgbm_models.joblib")
  print("\nTrained models saved to models/lgbm_models.joblib")

  # Post-process quantile crossing
  val_q10, val_q50, val_q90 = enforce_monotonicity(val_preds)
  test_q10, test_q50, test_q90 = enforce_monotonicity(test_preds)

  # Build Validation predictions DataFrame
  df_val_eval = splits["meta_val"].copy()
  df_val_eval["pred_point"] = np.maximum(0.0, val_preds["point"])
  df_val_eval["pred_q10"] = val_q10
  df_val_eval["pred_q50"] = val_q50
  df_val_eval["pred_q90"] = val_q90
  df_val_eval["split"] = "val"

  # Build Test predictions DataFrame
  df_test_eval = splits["meta_test"].copy()
  df_test_eval["pred_point"] = np.maximum(0.0, test_preds["point"])
  df_test_eval["pred_q10"] = test_q10
  df_test_eval["pred_q50"] = test_q50
  df_test_eval["pred_q90"] = test_q90
  # Baselines on test set: seasonal naive (lag 7) and rolling mean (rolling 28)
  df_test_eval["baseline_naive"] = splits["X_test"]["sales_lag_7"].values
  df_test_eval["baseline_moving_avg"] = splits["X_test"][
      "rolling_mean_28"
  ].values
  df_test_eval["split"] = "test"

  df_all_preds = pd.concat([df_val_eval, df_test_eval], ignore_index=True)
  df_all_preds.to_csv("data/processed/predictions.csv", index=False)
  print("Predictions saved to data/processed/predictions.csv")

  # Quick check on test set MAE
  y_test = splits["y_test"].values
  mae_point = mean_absolute_error(y_test, df_test_eval["pred_point"])
  mae_median = mean_absolute_error(y_test, test_q50)
  mae_naive = mean_absolute_error(y_test, df_test_eval["baseline_naive"])

  print("\n" + "=" * 50)
  print("INITIAL TEST ACCURACY SNAPSHOT (MAE):")
  print("=" * 50)
  print(f"Seasonal Naive Baseline MAE: {mae_naive:.3f}")
  print(f"Point Forecast Model MAE:    {mae_point:.3f}")
  print(f"Median Forecast (q50) MAE:   {mae_median:.3f}")
  print("=" * 50)


if __name__ == "__main__":
  run_pipeline()