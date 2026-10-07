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


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
  """Finite-sample split-conformal order statistic for a score vector."""
  scores = np.asarray(scores, dtype=float)
  if not len(scores):
    raise ValueError("Cannot calibrate an empty validation group.")
  rank = min(len(scores), int(np.ceil((1.0 - alpha) * (len(scores) + 1))))
  return float(np.sort(scores)[rank - 1])


def apply_cqr(
    df_preds: pd.DataFrame, alpha: float = 0.20
) -> tuple[pd.DataFrame, dict[str, float]]:
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

  # Use volume tiers derived strictly from training history. Each tier gets
  # its own validation correction; this lets interval width respond to the
  # demand scale without conditioning on the unknown future actual value.
  if "volume_tier" in df_val.columns and df_val["volume_tier"].notna().any():
    adjustments = {
        str(tier): _conformal_quantile(
            val_errors[df_val["volume_tier"].to_numpy() == tier], alpha
        )
        for tier in df_val["volume_tier"].dropna().unique()
    }
    fallback = _conformal_quantile(val_errors, alpha)
    df_test["cqr_adjustment"] = df_test["volume_tier"].map(adjustments).fillna(fallback)
  else:
    fallback = _conformal_quantile(val_errors, alpha)
    adjustments = {"All": fallback}
    df_test["cqr_adjustment"] = fallback

  delta = df_test["cqr_adjustment"]
  df_test["cqr_q10"] = np.maximum(0.0, df_test["pred_q10"] - delta)
  df_test["cqr_q90"] = np.maximum(0.0, df_test["pred_q90"] + delta)

  # 3. Baseline comparison: Residual-based interval on Point model
  residuals = df_val["sales"].values - df_val["pred_point"].values
  res_q10 = np.quantile(residuals, 0.10)
  res_q90 = np.quantile(residuals, 0.90)

  df_test["res_q10"] = np.maximum(0.0, df_test["pred_point"] + res_q10)
  df_test["res_q90"] = np.maximum(0.0, df_test["pred_point"] + res_q90)

  return df_test, adjustments


def run_calibration():
  preds_path = "data/processed/predictions.csv"
  print(f"Loading predictions from {preds_path}...")
  df_preds = pd.read_csv(preds_path)

  # Run CQR
  alpha = 0.20  # Nominal target: 80% coverage
  df_test_calibrated, adjustments = apply_cqr(df_preds, alpha=alpha)

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
  print("CQR adjustment by training-history volume tier:")
  for tier, adjustment in adjustments.items():
    print(f"  {tier}: {adjustment:+.3f} units")
  print("-" * 65)
  print(
      summary_df.to_string(
          index=False,
          formatters={
              "coverage": "{:.2%}".format,
              "mean_width": "{:.3f}".format,
              "winkler_score": "{:.3f}".format,
          },
      )
  )
  positive = df_test_calibrated["sales"] > 0
  print(
      "Positive-sales coverage: "
      f"{compute_interval_metrics(y_test[positive], df_test_calibrated.loc[positive, 'cqr_q10'].to_numpy(), df_test_calibrated.loc[positive, 'cqr_q90'].to_numpy(), alpha)['coverage']:.2%}"
  )
  print("=" * 65)

  # Save calibrated predictions for evaluation and inventory scripts
  out_path = "data/processed/predictions_calibrated.csv"
  df_test_calibrated.to_csv(out_path, index=False)
  print(f"\nCalibrated test dataset saved to {out_path}")


if __name__ == "__main__":
  run_calibration()
