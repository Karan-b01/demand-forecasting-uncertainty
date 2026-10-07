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
      merge_cols + [
          col for col in [
              "rolling_mean_28", "rolling_std_28", "is_weekend", "is_event",
              "price_ratio",
          ] if col in df_feat.columns
      ]
  ].drop_duplicates()
  df = df_preds.merge(feat_subset, on=merge_cols, how="left")

  df["date"] = pd.to_datetime(df["date"])
  return df


def audit_failure_regimes(df: pd.DataFrame) -> pd.DataFrame:
  """Report calibrated interval coverage on spikes, events, and price drops."""
  df = df.copy()

  # Define a demand spike relative to recent history when those features exist.
  if {"rolling_mean_28", "rolling_std_28"}.issubset(df.columns):
    threshold = df["rolling_mean_28"] + 2.0 * df["rolling_std_28"]
    df["is_spike"] = (df["sales"] > threshold).fillna(False).astype(int)
  else:
    df["is_spike"] = 0

  # 2. Base coverage boolean
  lower = "cqr_q10" if "cqr_q10" in df.columns else "pred_q10"
  upper = "cqr_q90" if "cqr_q90" in df.columns else "pred_q90"
  df["covered"] = (df["sales"] >= df[lower]) & (df["sales"] <= df[upper])
  df["interval_width"] = df[upper] - df[lower]

  regimes = {"All Test Records": df["sales"].notnull()}
  regimes["Zero Sales"] = df["sales"] == 0
  regimes["Positive Sales"] = df["sales"] > 0
  if {"rolling_mean_28", "rolling_std_28"}.issubset(df.columns):
    regimes["Usual Demand"] = df["is_spike"] == 0
    regimes["Demand Spikes (> 2 Std Dev)"] = df["is_spike"] == 1
  if "is_weekend" in df.columns:
    regimes["Weekdays"] = df["is_weekend"] == 0
    regimes["Weekends"] = df["is_weekend"] == 1
  if "is_event" in df.columns:
    event = df["is_event"].fillna(0).astype(bool)
    regimes["Regular Calendar Days"] = ~event
    regimes["Holiday or Event Days"] = event
  if "price_ratio" in df.columns:
    possible_promotion = df["price_ratio"] < 0.95
    regimes["Usual Price"] = ~possible_promotion.fillna(False)
    regimes["Lower Price (Possible Promotion)"] = possible_promotion.fillna(False)

  summary = []
  for label, mask in regimes.items():
    sub = df[mask]
    summary.append({
        "Regime": label,
        "Sample Count": len(sub),
        "80% Interval Coverage": sub["covered"].mean() if len(sub) else np.nan,
        "Mean Interval Width": sub["interval_width"].mean() if len(sub) else np.nan,
        "Mean Actual Sales": sub["sales"].mean() if len(sub) else np.nan,
    })

  return pd.DataFrame(summary), df


def compare_interval_methods(df: pd.DataFrame) -> pd.DataFrame:
  """Compare raw quantile, CQR, and residual intervals across key regimes."""
  tagged = df.copy()
  if "is_spike" not in tagged:
    if {"rolling_mean_28", "rolling_std_28"}.issubset(tagged.columns):
      tagged["is_spike"] = tagged.sales.gt(
          tagged.rolling_mean_28 + 2 * tagged.rolling_std_28
      ).fillna(False)
    else:
      tagged["is_spike"] = False
  masks = {
      "All test days": pd.Series(True, index=tagged.index),
      "Zero-sales days": tagged.sales.eq(0),
      "Positive-sales days": tagged.sales.gt(0),
      "Demand spikes": tagged.is_spike.astype(bool),
  }
  if "volume_tier" in tagged:
    for tier in ("Low", "Medium", "High"):
      masks[f"{tier} historical-volume tier"] = tagged.volume_tier.eq(tier)

  methods = [
      ("Raw quantile", "pred_q10", "pred_q90"),
      ("CQR calibrated", "cqr_q10", "cqr_q90"),
      ("Residual-based", "res_q10", "res_q90"),
  ]
  rows = []
  for method, lower, upper in methods:
    if not {lower, upper}.issubset(tagged.columns):
      continue
    tagged[f"{method}_covered"] = (
        tagged.sales.ge(tagged[lower]) & tagged.sales.le(tagged[upper])
    )
    tagged[f"{method}_width"] = tagged[upper] - tagged[lower]
    for group, mask in masks.items():
      part = tagged.loc[mask]
      rows.append({
          "Method": method,
          "Test slice": group,
          "Records": len(part),
          "Coverage": part[f"{method}_covered"].mean() if len(part) else np.nan,
          "Mean width": part[f"{method}_width"].mean() if len(part) else np.nan,
      })
  return pd.DataFrame(rows)


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
  lower = "cqr_q10" if "cqr_q10" in sub.columns else "pred_q10"
  upper = "cqr_q90" if "cqr_q90" in sub.columns else "pred_q90"
  ax.fill_between(
      sub["date"],
      sub[lower],
      sub[upper],
      color="#d62728",
      alpha=0.2,
      label="Calibrated 80% Prediction Interval",
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

  method_comparison = compare_interval_methods(df_tagged)
  print("\nINTERVAL METHOD COMPARISON BY DEMAND SLICE")
  print(method_comparison.to_string(
      index=False,
      formatters={"Coverage": "{:.1%}".format, "Mean width": "{:.2f}".format, "Records": "{:,}".format},
  ))
  os.makedirs("reports", exist_ok=True)
  method_comparison.to_csv("reports/interval_method_comparison.csv", index=False)
  print("Saved reports/interval_method_comparison.csv")

  plot_failure_modes(df_tagged)


if __name__ == "__main__":
  main()
