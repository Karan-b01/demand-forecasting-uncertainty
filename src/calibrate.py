import numpy as np
import pandas as pd


def compute_interval_metrics(
    y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, alpha: float = 0.20
) -> dict:
  """Calculates empirical coverage, mean interval width, and Winkler Score.

  Winkler score rewards sharpness (narrow intervals) while heavily penalizing
  unrealized boundary violations.
  """
  coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
  width = np.mean(y_upper - y_lower)

  underage = np.maximum(0.0, y_lower - y_true)
  overage = np.maximum(0.0, y_true - y_upper)
  winkler = (y_upper - y_lower) + (2.0 / alpha) * (underage + overage)

  return {
      "coverage": float(coverage),
      "mean_width": float(width),
      "winkler_score": float(np.mean(winkler)),
  }


def apply_cqr(
    df_preds: pd.DataFrame, alpha: float = 0.20
) -> tuple[pd.DataFrame, float]:
  """Computes Conformalized Quantile Regression (CQR) adjustment on validation set

  and calibrates test predictions.
  """
  val_mask = df_preds["split"] == "val"
  test_mask = df_preds["split"] == "test"

  df_val = df_preds[val_mask].copy()
  df_test = df_preds[test_mask].copy()

  # 1. Nonconformity score on calibration (validation) set
  y_val = df_val["sales"].values
  q10_val = df_val["pred_q10"].values
  q90_val = df_val["pred_q90"].values

  # Nonconformity error: positive if y is outside [q10, q90]
  val_errors = np.maximum(q10_val - y_val, y_val - q90_val)

  # Conformal correction factor with finite-sample adjustment
  n_val = len(val_errors)
  q_level = np.clip(
      np.ceil((1.0 - alpha) * (n_val + 1)) / n_val, 0.0, 1.0
  )
  conformal_adjustment = float(
      np.quantile(val_errors, q_level, method="higher")
  )

  # 2. Adjust test intervals
  df_test["cqr_q10"] = np.maximum(
      0.0, df_test["pred_q10"] - conformal_adjustment
  )
  df_test["cqr_q90"] = np.maximum(
      0.0, df_test["pred_q90"] + conformal_adjustment
  )

  # 3. Baseline comparison: Residual-based interval on Point model
  residuals = df_val["sales"].values - df_val["pred_point"].values
  res_q10 = np.quantile(residuals, 0.10)
  res_q90 = np.quantile(residuals, 0.90)

  df_test["res_q10"] = np.maximum(0.0, df_test["pred_point"] + res_q10)
  df_test["res_q90"] = np.maximum(0.0, df_test["pred_point"] + res_q90)

  return df_test, conformal_adjustment


def run_calibration():
  preds_path = "data/processed/predictions.csv"
  print(f"Loading predictions from {preds_path}...")
  df_preds = pd.read_csv(preds_path)

  # Run CQR
  alpha = 0.20  # Nominal target: 80% coverage
  df_test_calibrated, adjustment = apply_cqr(df_preds, alpha=alpha)

  y_test = df_test_calibrated["sales"].values

  # Compare interval strategies
  raw_metrics = compute_interval_metrics(
      y_test,
      df_test_calibrated["pred_q10"].values,
      df_test_calibrated["pred_q90"].values,
      alpha,
  )

  cqr_metrics = compute_interval_metrics(
      y_test,
      df_test_calibrated["cqr_q10"].values,
      df_test_calibrated["cqr_q90"].values,
      alpha,
  )

  res_metrics = compute_interval_metrics(
      y_test,
      df_test_calibrated["res_q10"].values,
      df_test_calibrated["res_q90"].values,
      alpha,
  )

  # Summary Table
  summary_df = pd.DataFrame([
      {"Method": "Residual-Based Interval", **res_metrics},
      {"Method": "Raw LightGBM Quantile", **raw_metrics},
      {"Method": "Conformalized (CQR)", **cqr_metrics},
  ])

  print("\n" + "=" * 65)
  print(f"INTERVAL CALIBRATION RESULTS (Nominal Target: {int((1-alpha)*100)}%)")
  print("=" * 65)
  print(f"CQR Adjustment Scalar Added to Margins: +{adjustment:.3f} units")
  print("-" * 65)
  print(
      summary_df.to_string(
          index=False, formatters={"coverage": "{:.2%}".format}
      )
  )
  print("=" * 65)

  # Save calibrated predictions for evaluation and inventory scripts
  out_path = "data/processed/predictions_calibrated.csv"
  df_test_calibrated.to_csv(out_path, index=False)
  print(f"\nCalibrated test dataset saved to {out_path}")


if __name__ == "__main__":
  run_calibration()