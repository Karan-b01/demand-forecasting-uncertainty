import numpy as np
import pandas as pd

from src.calibrate import apply_cqr, compute_interval_metrics


def test_interval_metrics_measure_inclusive_coverage_and_width():
  result = compute_interval_metrics(
      np.array([0.0, 2.0, 5.0]),
      np.array([0.0, 1.0, 1.0]),
      np.array([1.0, 3.0, 4.0]),
  )
  assert result["coverage"] == 2 / 3
  assert result["mean_width"] == 2.0


def test_cqr_uses_validation_only_and_calibrates_volume_tiers():
  validation = pd.DataFrame({
      "split": ["val"] * 10,
      "sales": [1.0] * 5 + [6.0] * 5,
      "pred_q10": [0.0] * 10,
      "pred_q50": [1.0] * 10,
      "pred_q90": [2.0] * 5 + [1.0] * 5,
      "pred_point": [1.0] * 10,
      "volume_tier": ["Low"] * 5 + ["High"] * 5,
  })
  test = pd.DataFrame({
      "split": ["test", "test"],
      "sales": [1000.0, 1000.0],  # Must not affect calibration scores.
      "pred_q10": [0.0, 0.0],
      "pred_q50": [1.0, 1.0],
      "pred_q90": [2.0, 1.0],
      "pred_point": [1.0, 1.0],
      "volume_tier": ["Low", "High"],
  })

  calibrated, adjustments = apply_cqr(pd.concat([validation, test]), alpha=0.2)

  assert set(adjustments) == {"Low", "High"}
  assert adjustments["Low"] == -1.0
  assert adjustments["High"] == 5.0
  assert calibrated["cqr_q10"].tolist() == [1.0, 0.0]
  assert calibrated["cqr_q90"].tolist() == [1.0, 6.0]
