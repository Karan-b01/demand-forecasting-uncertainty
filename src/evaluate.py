import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def load_data(
    pred_path: str = "data/processed/predictions_calibrated.csv",
) -> pd.DataFrame:
  if not os.path.exists(pred_path):
    raise FileNotFoundError(f"Missing {pred_path}. Run src.calibrate first.")
  df = pd.read_csv(pred_path)
  df["date"] = pd.to_datetime(df["date"])
  return df


def evaluate_point_forecasts(df: pd.DataFrame) -> pd.DataFrame:
  """Computes MAE and RMSE across all models and simple baselines."""
  y = df["sales"].values
  metrics = []

  models = {
      "Seasonal Naive (Lag 7)": df["baseline_naive"].values,
      "Moving Average (28d Mean)": df["baseline_moving_avg"].values,
      "Point Forecast (L2/RMSE)": df["pred_point"].values,
      "Median Forecast (q50)": df["pred_q50"].values,
  }

  for name, preds in models.items():
    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)
    metrics.append({"Model": name, "MAE": mae, "RMSE": rmse})

  return pd.DataFrame(metrics)


def evaluate_segment_coverage(df: pd.DataFrame) -> pd.DataFrame:
  """Audits empirical 80% coverage across item volume tiers and day of week."""
  df = df.copy()
  df["covered"] = (df["sales"] >= df["pred_q10"]) & (
      df["sales"] <= df["pred_q90"]
  )
  df["day_name"] = df["date"].dt.day_name()

  # Volume tiers by item
  item_volume = (
      df.groupby("item")["sales"].mean().rename("mean_item_sales").reset_index()
  )
  df = df.merge(item_volume, on="item", how="left")
  df["volume_tier"] = pd.qcut(
      df["mean_item_sales"], q=3, labels=["Low Volume", "Mid Volume", "High Volume"]
  )

  # Segment summaries
  tier_summary = (
      df.groupby("volume_tier", observed=False)["covered"]
      .agg(["count", "mean"])
      .reset_index()
  )
  tier_summary.columns = ["Segment", "Count", "Coverage"]

  day_summary = (
      df.groupby("day_name", observed=False)["covered"]
      .agg(["count", "mean"])
      .reindex([
          "Monday",
          "Tuesday",
          "Wednesday",
          "Thursday",
          "Friday",
          "Saturday",
          "Sunday",
      ])
      .reset_index()
  )
  day_summary.columns = ["Segment", "Count", "Coverage"]

  return pd.concat([tier_summary, day_summary], ignore_index=True)


def plot_reliability_diagram(
    df: pd.DataFrame, output_dir: str = "reports/figures"
):
  """Plots nominal quantile level vs empirical fraction below predictions."""
  os.makedirs(output_dir, exist_ok=True)
  y = df["sales"].values

  # Calculate empirical fraction below each predicted quantile
  nom_quantiles = [0.10, 0.50, 0.90]
  obs_fractions = [
      np.mean(y <= df["pred_q10"].values),
      np.mean(y <= df["pred_q50"].values),
      np.mean(y <= df["pred_q90"].values),
  ]

  plt.figure(figsize=(6, 6))
  plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
  plt.plot(
      nom_quantiles,
      obs_fractions,
      marker="o",
      linewidth=2,
      color="#1f77b4",
      label="LightGBM Quantiles",
  )

  for x, y_pt in zip(nom_quantiles, obs_fractions):
    plt.annotate(
        f"{y_pt:.1%}",
        (x, y_pt),
        textcoords="offset points",
        xytext=(-15, 10),
        fontweight="bold",
    )

  plt.title("Reliability Diagram (Quantile Calibration)", fontsize=13, pad=12)
  plt.xlabel("Nominal Quantile Level", fontsize=11)
  plt.ylabel("Observed Empirical Proportion", fontsize=11)
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.legend(loc="upper left")
  plt.tight_layout()

  save_path = os.path.join(output_dir, "reliability_diagram.png")
  plt.savefig(save_path, dpi=300)
  plt.close()
  print(f"Reliability chart saved: {save_path}")


def plot_interval_fan_chart(
    df: pd.DataFrame,
    sample_item: str = None,
    output_dir: str = "reports/figures",
):
  """Plots a representative demand forecast envelope against ground truth."""
  os.makedirs(output_dir, exist_ok=True)

  if sample_item is None:
    sample_item = df["item"].iloc[0]

  sub = (
      df[df["item"] == sample_item].sort_values("date").reset_index(drop=True)
  )

  plt.figure(figsize=(11, 5))
  plt.plot(
      sub["date"],
      sub["sales"],
      color="black",
      marker="o",
      markersize=3,
      label="Actual Demand",
      linewidth=1.2,
  )
  plt.plot(
      sub["date"],
      sub["pred_q50"],
      color="#1f77b4",
      linestyle="--",
      label="Median (P50)",
      linewidth=1.5,
  )
  plt.fill_between(
      sub["date"],
      sub["pred_q10"],
      sub["pred_q90"],
      color="#1f77b4",
      alpha=0.25,
      label="80% Prediction Interval (P10 - P90)",
  )

  plt.title(
      f"Uncertainty Forecast Envelope vs Actuals: {sample_item}",
      fontsize=13,
      pad=12,
  )
  plt.xlabel("Date", fontsize=11)
  plt.ylabel("Sales Volume", fontsize=11)
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.legend(loc="upper left")
  plt.tight_layout()

  save_path = os.path.join(output_dir, "forecast_envelope.png")
  plt.savefig(save_path, dpi=300)
  plt.close()
  print(f"Envelope plot saved: {save_path}")


def run_evaluation():
  df = load_data()

  print("=" * 65)
  print("1. POINT FORECAST BENCHMARK EVALUATION (TEST SET)")
  print("=" * 65)
  df_point = evaluate_point_forecasts(df)
  print(df_point.to_string(index=False))

  print("\n" + "=" * 65)
  print("2. 80% PREDICTION INTERVAL AUDIT BY SEGMENT")
  print("=" * 65)
  df_seg = evaluate_segment_coverage(df)
  print(
      df_seg.to_string(
          index=False,
          formatters={"Coverage": "{:.1%}".format, "Count": "{:,}".format},
      )
  )

  print("\n" + "=" * 65)
  print("3. EXPORTING EVALUATION DIAGNOSTICS")
  print("=" * 65)
  plot_reliability_diagram(df)
  plot_interval_fan_chart(df)


if __name__ == "__main__":
  run_evaluation()