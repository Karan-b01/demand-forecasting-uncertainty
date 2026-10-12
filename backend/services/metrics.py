"""Forecast evaluation and interval summary calculations."""

import numpy as np


def forecast_metrics(df):
    y = df.sales.to_numpy(float)
    median = df.pred_q50.to_numpy(float)
    point = df.pred_point.to_numpy(float)
    lower = df.get("cqr_q10", df.pred_q10).to_numpy(float)
    upper = df.get("cqr_q90", df.pred_q90).to_numpy(float)
    raw_lower, raw_upper = df.pred_q10.to_numpy(float), df.pred_q90.to_numpy(float)
    covered = (y >= lower) & (y <= upper)
    positive, zero = y > 0, y == 0
    result = {
        "rows": int(len(df)),
        "mae": float(np.mean(abs(y - median))),
        "rmse": float(np.sqrt(np.mean((y - median) ** 2))),
        "point_mae": float(np.mean(abs(y - point))),
        "point_rmse": float(np.sqrt(np.mean((y - point) ** 2))),
        "coverage": float(np.mean(covered)),
        "interval_width": float(np.mean(upper - lower)),
        "positive_coverage": (
            float(np.mean(covered[positive])) if positive.any() else None
        ),
        "positive_rows": int(positive.sum()),
        "zero_coverage": float(np.mean(covered[zero])) if zero.any() else None,
        "zero_rows": int(zero.sum()),
        "raw_coverage": float(np.mean((y >= raw_lower) & (y <= raw_upper))),
        "raw_width": float(np.mean(raw_upper - raw_lower)),
        "calibration_expansion": (
            float(np.mean(upper - raw_upper)) if "cqr_q90" in df else 0.0
        ),
        "p10_below_strict": float(np.mean(y < df.pred_q10)),
        "p10_below_or_equal": float(np.mean(y <= df.pred_q10)),
        "p50_below_strict": float(np.mean(y < df.pred_q50)),
        "p50_below_or_equal": float(np.mean(y <= df.pred_q50)),
        "p90_below_strict": float(np.mean(y < df.pred_q90)),
        "p90_below_or_equal": float(np.mean(y <= df.pred_q90)),
    }
    for column, prefix in (
        ("baseline_moving_avg", "moving_average"),
        ("baseline_naive", "weekly_repeat"),
    ):
        if column in df:
            baseline = df[column].to_numpy(float)
            result[f"{prefix}_mae"] = float(np.mean(abs(y - baseline)))
            result[f"{prefix}_rmse"] = float(np.sqrt(np.mean((y - baseline) ** 2)))
    return result


def interval_summary(df, lower, upper, label):
    y = df.sales.to_numpy(float)
    lo, hi = df[lower].to_numpy(float), df[upper].to_numpy(float)
    return {
        "method": label,
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "width": float(np.mean(hi - lo)),
    }
