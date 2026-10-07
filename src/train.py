import os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


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
      "sales",
      "sell_price",
  ]
  feature_cols = [c for c in df.columns if c not in ignore_cols]

  # Ensure categorical columns are properly typed for LightGBM
  categorical_cols = [
      "store", "item", "state_id", "dept_id", "cat_id",
      "event_name_1", "event_type_1", "event_name_2", "event_type_2",
  ]
  for col in categorical_cols:
    if col in feature_cols:
      df[col] = df[col].astype("category")

  # Split dates: Final 28 days = Test, prior 28 days = Validation
  max_date = df["date"].max()
  test_start = max_date - pd.Timedelta(days=27)
  val_start = test_start - pd.Timedelta(days=28)

  train_mask = df["date"] < val_start
  val_mask = (df["date"] >= val_start) & (df["date"] < test_start)
  test_mask = df["date"] >= test_start

  # Build volume tiers only from training-period targets. The resulting tier
  # is known for a series before validation/test and can safely stratify CQR.
  series_keys = ["store", "item"]
  series_volume = (
      df.loc[train_mask].groupby(series_keys, observed=True)[target_col]
      .mean()
      .rename("historical_mean_sales")
      .reset_index()
  )
  series_volume["volume_rank"] = series_volume["historical_mean_sales"].rank(
      method="first", pct=True
  )
  series_volume["volume_tier"] = pd.cut(
      series_volume["volume_rank"],
      bins=[0.0, 1 / 3, 2 / 3, 1.0],
      labels=["Low", "Medium", "High"],
      include_lowest=True,
  ).astype(str)
  df = df.merge(
      series_volume[series_keys + ["historical_mean_sales", "volume_tier"]],
      on=series_keys,
      how="left",
      validate="many_to_one",
  )

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
      "meta_train": df.loc[
          train_mask,
          ["date", "store", "item", "sales", "historical_mean_sales", "volume_tier"],
      ],
      "meta_val": df.loc[
          val_mask,
          ["date", "store", "item", "sales", "historical_mean_sales", "volume_tier"],
      ],
      "meta_test": df.loc[
          test_mask,
          ["date", "store", "item", "sales", "historical_mean_sales", "volume_tier"],
      ],
      "feature_cols": feature_cols,
  }
  return splits


def train_models(splits: dict):
  """Trains point and quantile models, including upper-tail inventory levels."""
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
  quantiles = [0.1, 0.5, 0.75, 0.9, 0.95, 0.99]
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

  # A pooled P10 model can collapse toward zero for a high-volume series when
  # its training target mixes intermittent and high-volume products. Compare
  # it with a lagged empirical lower-tail feature on validation, then keep the
  # blend weight that minimizes validation pinball loss for each history tier.
  val_meta = splits["meta_val"]
  test_meta = splits["meta_test"]
  raw_val_p10 = val_preds["q_0.1"].copy()
  raw_test_p10 = test_preds["q_0.1"].copy()
  val_p10 = raw_val_p10.copy()
  test_p10 = raw_test_p10.copy()
  blend_weights = {}
  q = 0.10
  for tier in ["Low", "Medium", "High"]:
    val_mask = val_meta["volume_tier"].eq(tier).to_numpy()
    test_mask = test_meta["volume_tier"].eq(tier).to_numpy()
    if not val_mask.any() or not test_mask.any():
      continue
    local_val = X_val.loc[val_mask, "rolling_q10_56"].to_numpy(float)
    local_test = X_test.loc[test_mask, "rolling_q10_56"].to_numpy(float)
    y_tier = y_val.loc[val_mask].to_numpy(float)
    candidate_losses = {}
    for local_weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
      candidate = (1.0 - local_weight) * raw_val_p10[val_mask] + local_weight * local_val
      error = y_tier - candidate
      candidate_losses[local_weight] = float(
          np.maximum(q * error, (q - 1.0) * error).mean()
      )
    weight = min(candidate_losses, key=candidate_losses.get)
    blend_weights[tier] = weight
    val_p10[val_mask] = (
        (1.0 - weight) * raw_val_p10[val_mask] + weight * local_val
    )
    test_p10[test_mask] = (
        (1.0 - weight) * raw_test_p10[test_mask] + weight * local_test
    )
  val_preds["q_0.1"] = val_p10
  test_preds["q_0.1"] = test_p10
  val_preds["q10_model_raw"] = raw_val_p10
  test_preds["q10_model_raw"] = raw_test_p10
  models["p10_validation_blend_weights"] = blend_weights

  return models, val_preds, test_preds


def enforce_monotonicity(
    preds_dict: dict, prefix: str = ""
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return sorted non-crossing outputs for all trained quantile levels."""
  levels = sorted(
      (float(key.removeprefix("q_")), key)
      for key in preds_dict
      if key.startswith("q_")
  )
  stacked = np.maximum(0.0, np.vstack([preds_dict[key] for _, key in levels]))
  stacked = np.sort(stacked, axis=0)
  return {f"pred_q{int(level * 100):02d}": stacked[i] for i, (level, _) in enumerate(levels)}


def run_pipeline():
  os.makedirs("models", exist_ok=True)
  os.makedirs("data/processed", exist_ok=True)

  splits = prepare_data_splits()
  models, val_preds, test_preds = train_models(splits)

  # Save trained models
  joblib.dump(models, "models/lgbm_models.joblib")
  print("\nTrained models saved to models/lgbm_models.joblib")

  # Post-process quantile crossing
  val_quantiles = enforce_monotonicity(val_preds)
  test_quantiles = enforce_monotonicity(test_preds)

  # Build Validation predictions DataFrame
  df_val_eval = splits["meta_val"].copy()
  df_val_eval["pred_point"] = np.maximum(0.0, val_preds["point"])
  for name, values in val_quantiles.items():
    df_val_eval[name] = values
  df_val_eval["pred_q10_model"] = val_preds["q10_model_raw"]
  df_val_eval["split"] = "val"

  # Build Test predictions DataFrame
  df_test_eval = splits["meta_test"].copy()
  df_test_eval["pred_point"] = np.maximum(0.0, test_preds["point"])
  for name, values in test_quantiles.items():
    df_test_eval[name] = values
  df_test_eval["pred_q10_model"] = test_preds["q10_model_raw"]
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
  mae_median = mean_absolute_error(y_test, test_quantiles["pred_q50"])
  mae_naive = mean_absolute_error(y_test, df_test_eval["baseline_naive"])
  rmse_point = mean_squared_error(y_test, df_test_eval["pred_point"]) ** 0.5

  print("\n" + "=" * 50)
  print("INITIAL TEST ACCURACY SNAPSHOT (MAE):")
  print("=" * 50)
  print(f"Seasonal Naive Baseline MAE: {mae_naive:.3f}")
  print(f"Point Forecast Model MAE:    {mae_point:.3f}")
  print(f"Median Forecast (q50) MAE:   {mae_median:.3f}")
  print(f"Point Forecast Model RMSE:   {rmse_point:.3f}")
  print("=" * 50)


if __name__ == "__main__":
  run_pipeline()
