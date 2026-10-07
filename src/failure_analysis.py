import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_full_test_data() -> pd.DataFrame:
  preds_path = "data/processed/predictions_calibrated.csv"
  features_path = "data/processed/features.csv"

  df_preds = pd.read_csv(preds_path)
  df_feat = pd.read_csv(features_path)

  # Merge rolling statistics to identify demand spikes
  merge_cols = ["date", "store", "item"]
  feat_subset = df_feat[
      merge_cols + ["rolling_mean_28", "rolling_std_28", "is_weekend"]
  ].drop_duplicates()
  df = df_preds.merge(feat_subset, on=merge_cols, how="left")

  df["date"] = pd.to_datetime(df["date"])
  return df


def audit_failure_regimes(df: pd.DataFrame) -> pd.DataFrame:
  """Tags observations into distinct operational regimes and checks 80% coverage."""
  df = df.copy()

  # 1. Define Demand Spike: actual demand exceeds rolling historical mean by > 2 standard deviations
  threshold = df["rolling_mean_28"] + 2.0 * df["rolling_std_28"]
  df["is_spike"] = (df["sales"] > threshold).astype(int)

  # 2. Base coverage boolean
  df["covered_raw"] = (df["sales"] >= df["pred_q10"]) & (
      df["sales"] <= df["pred_q90"]
  )
  df["interval_width"] = df["pred_q90"] - df["pred_q10"]

  regimes = {
      "All Test Records": df["sales"].notnull(),
      "Normal Regimes (Non-Spike)": df["is_spike"] == 0,
      "Demand Spikes (> 2 Std Dev)": df["is_spike"] == 1,
      "Weekdays": df["is_weekend"] == 0,
      "Weekends": df["is_weekend"] == 1,
  }

  summary = []
  for label, mask in regimes.items():
    sub = df[mask]
    if len(sub) == 0:
      continue
    summary.append({
        "Regime": label,
        "Sample Count": len(sub),
        "80% Interval Coverage": sub["covered_raw"].mean(),
        "Mean Interval Width": sub["interval_width"].mean(),
        "Mean Actual Sales": sub["sales"].mean(),
    })

  return pd.DataFrame(summary), df


def plot_failure_modes(df: pd.DataFrame, output_dir: str = "reports/figures"):
  os.makedirs(output_dir, exist_ok=True)

  # Find an item that experienced a spike during the test period
  spiked_items = df[df["is_spike"] == 1]["item"].unique()
  chosen_item = spiked_items[0] if len(spiked_items) > 0 else df["item"].iloc[0]

  sub = df[df["item"] == chosen_item].sort_values("date").reset_index(drop=True)

  fig, ax = plt.subplots(figsize=(10, 4.5))

  ax.plot(
      sub["date"],
      sub["sales"],
      color="black",
      marker="o",
      label="Actual Demand",
      linewidth=1.2,
  )
  ax.fill_between(
      sub["date"],
      sub["pred_q10"],
      sub["pred_q90"],
      color="#d62728",
      alpha=0.2,
      label="Raw 80% Quantile Interval",
  )

  # Highlight spikes
  spikes = sub[sub["is_spike"] == 1]
  if not spikes.empty:
    ax.scatter(
        spikes["date"],
        spikes["sales"],
        color="red",
        s=90,
        zorder=5,
        label="Detected Spikes (> 2σ)",
    )

  ax.set_title(
      f"Failure Mode Inspection: Interval Collapse During Spikes ({chosen_item})",
      fontsize=12,
  )
  ax.set_ylabel("Daily Sales Units")
  ax.set_xlabel("Date")
  ax.grid(True, linestyle=":", alpha=0.6)
  ax.legend(loc="upper left")
  plt.tight_layout()

  save_path = os.path.join(output_dir, "failure_analysis_spikes.png")
  plt.savefig(save_path, dpi=300)
  plt.close()
  print(f"Spike diagnostic figure saved: {save_path}")


def main():
  df = load_full_test_data()
  summary_df, df_tagged = audit_failure_regimes(df)

  print("=" * 75)
  print("REGIME-CONDITIONAL FAILURE ANALYSIS (Nominal Target: 80.0%)")
  print("=" * 75)
  print(
      summary_df.to_string(
          index=False,
          formatters={
              "Sample Count": "{:,}".format,
              "80% Interval Coverage": "{:.1%}".format,
              "Mean Interval Width": "{:.2f}".format,
              "Mean Actual Sales": "{:.2f}".format,
          },
      )
  )

  plot_failure_modes(df_tagged)


if __name__ == "__main__":
  main()