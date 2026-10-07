import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_test_predictions(
    pred_path: str = "data/processed/predictions_calibrated.csv",
) -> pd.DataFrame:
  if not os.path.exists(pred_path):
    raise FileNotFoundError(
        f"Missing {pred_path}. Run src.calibrate and src.evaluate first."
    )
  df = pd.read_csv(pred_path)
  df["date"] = pd.to_datetime(df["date"])
  return df


def simulate_inventory_cost(
    demand: np.ndarray, stock: np.ndarray, c_u: float = 3.0, c_o: float = 1.0
) -> dict:
  """Calculates total financial loss, stockout frequency, and excess units."""
  underage_units = np.maximum(0.0, demand - stock)
  overage_units = np.maximum(0.0, stock - demand)

  total_underage_cost = c_u * np.sum(underage_units)
  total_overage_cost = c_o * np.sum(overage_units)
  total_cost = total_underage_cost + total_overage_cost

  service_level = np.mean(stock >= demand)
  stockout_rate = np.mean(demand > stock)

  return {
      "Total Cost ($)": total_cost,
      "Underage Cost ($)": total_underage_cost,
      "Overage Cost ($)": total_overage_cost,
      "Service Level": service_level,
      "Stockout Rate": stockout_rate,
      "Mean Stock": np.mean(stock),
  }


def forecast_at_quantile(df: pd.DataFrame, tau: float) -> np.ndarray:
  """Interpolate between available trained quantiles for each forecast row."""
  level_cols = [
      (level / 100, f"pred_q{level:02d}")
      for level in (10, 50, 75, 90, 95, 99)
      if f"pred_q{level:02d}" in df
  ]
  if not level_cols:
    raise ValueError("No trained quantile forecast columns are available.")
  levels = np.array([level for level, _ in level_cols])
  values = np.column_stack([df[col].to_numpy(float) for _, col in level_cols])
  if tau <= levels[0]:
    return np.maximum(0.0, values[:, 0])
  if tau >= levels[-1]:
    return np.maximum(0.0, values[:, -1])
  upper = int(np.searchsorted(levels, tau, side="right"))
  lower = upper - 1
  weight = (tau - levels[lower]) / (levels[upper] - levels[lower])
  return np.maximum(0.0, values[:, lower] + weight * (values[:, upper] - values[:, lower]))


def run_strategy_comparison(
    df: pd.DataFrame, c_u: float = 3.0, c_o: float = 1.0
) -> pd.DataFrame:
  """Compares stocking policies against test actuals."""
  demand = df["sales"].values

  # Strategy 1: Stocking at Point Forecast (RMSE model)
  stock_point = df["pred_point"].values

  # Strategy 2: Stocking at Median (q50)
  stock_median = df["pred_q50"].values

  # Quantile Newsvendor Policy
  # Critical ratio tau* = 3 / (3 + 1) = 0.75
  # Linear interpolation between q50 and q90
  # tau* is 62.5% of the distance from 0.50 to 0.90
  stock_quantile_75 = forecast_at_quantile(df, 0.75)

  # Conservative upper-band policy
  stock_q90 = df["pred_q90"].values

  strategies = {
      "1. Point Forecast (RMSE)": stock_point,
      "2. Median Forecast (q50)": stock_median,
      "3. Optimal Quantile Policy (tau*=0.75)": stock_quantile_75,
      "4. Conservative P90": stock_q90,
  }

  results = []
  for name, stock in strategies.items():
    metrics = simulate_inventory_cost(demand, stock, c_u=c_u, c_o=c_o)
    results.append({"Strategy": name, **metrics})

  return pd.DataFrame(results)


def run_cost_sensitivity(
    df: pd.DataFrame, output_dir: str = "reports/figures"
):
  """Evaluates total cost saved by Quantile Policy vs Point Policy across Cu/Co ratios."""
  os.makedirs(output_dir, exist_ok=True)
  demand = df["sales"].values

  ratios = [1.0, 2.0, 3.0, 5.0]
  c_o = 1.0

  comparison_data = []
  for ratio in ratios:
    c_u = ratio * c_o
    tau_star = c_u / (c_u + c_o)

    stock_opt = forecast_at_quantile(df, tau_star)

    cost_point = simulate_inventory_cost(
        demand, df["pred_point"].values, c_u=c_u, c_o=c_o
    )["Total Cost ($)"]
    cost_opt = simulate_inventory_cost(demand, stock_opt, c_u=c_u, c_o=c_o)[
        "Total Cost ($)"
    ]

    savings_pct = (cost_point - cost_opt) / cost_point * 100.0
    comparison_data.append({
        "Cu/Co Ratio": f"{int(ratio)}:1",
        "Critical Fractile (tau*)": f"{tau_star:.2%}",
        "Point Cost ($)": cost_point,
        "Quantile Cost ($)": cost_opt,
        "Savings (%)": savings_pct,
    })

  summary_df = pd.DataFrame(comparison_data)

  # Plotting sensitivity curve
  plt.figure(figsize=(7, 4.5))
  plt.plot(
      [1.0, 2.0, 3.0, 5.0],
      summary_df["Savings (%)"],
      marker="s",
      color="#2ca02c",
      linewidth=2.2,
  )
  plt.title(
      "Inventory Savings of Quantile Stocking vs. Point Forecast",
      fontsize=12,
      pad=12,
  )
  plt.xlabel("Underage-to-Overage Cost Ratio (Cu / Co)", fontsize=11)
  plt.ylabel("Cost Reduction (%)", fontsize=11)
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.tight_layout()

  save_path = os.path.join(output_dir, "inventory_cost_sensitivity.png")
  plt.savefig(save_path, dpi=300)
  plt.close()

  return summary_df


def run_stock_quantile_cost_curve(
    df: pd.DataFrame, output_dir: str = "reports/figures"
) -> pd.DataFrame:
  """Compare test-set cost across stock quantiles and shortage/leftover costs."""
  os.makedirs(output_dir, exist_ok=True)
  levels = [0.10, 0.50, 0.75, 0.90, 0.95, 0.99]
  columns = [f"pred_q{int(level * 100):02d}" for level in levels]
  available = [(level, col) for level, col in zip(levels, columns) if col in df]
  if len(available) < 3:
    raise ValueError("Cost curve requires P10/P50/P90, and ideally P95/P99 forecasts.")

  grid = np.round(np.arange(0.50, 0.991, 0.01), 2)
  level_values = np.array([level for level, _ in available], dtype=float)
  prediction_values = np.column_stack([
      np.maximum(0.0, df[col].to_numpy(float)) for _, col in available
  ])
  ratios = [1, 2, 3, 5]
  records = []
  demand = df["sales"].to_numpy(float)
  for ratio in ratios:
    c_u, c_o = float(ratio), 1.0
    for tau in grid:
      upper_idx = min(
          max(int(np.searchsorted(level_values, tau, side="right")), 1),
          len(level_values) - 1,
      )
      lower_idx = upper_idx - 1
      weight = (tau - level_values[lower_idx]) / (
          level_values[upper_idx] - level_values[lower_idx]
      )
      stock = prediction_values[:, lower_idx] + weight * (
          prediction_values[:, upper_idx] - prediction_values[:, lower_idx]
      )
      cost = simulate_inventory_cost(demand, stock, c_u=c_u, c_o=c_o)["Total Cost ($)"]
      records.append({
          "Stock Quantile": tau,
          "Cu/Co Ratio": f"{ratio}:1",
          "Total Cost ($)": cost,
          "Theoretical Target": ratio / (ratio + 1),
      })

  curve = pd.DataFrame(records)
  fig, ax = plt.subplots(figsize=(8.5, 5.0))
  colors = {"1:1": "#667085", "2:1": "#2f80ed", "3:1": "#159a72", "5:1": "#e07a35"}
  for label, group in curve.groupby("Cu/Co Ratio", sort=False):
    ax.plot(group["Stock Quantile"], group["Total Cost ($)"], label=f"{label} shortage:extra", color=colors[label], linewidth=2.2)
    target = float(group["Theoretical Target"].iloc[0])
    ax.axvline(target, color=colors[label], linestyle=":", alpha=0.35, linewidth=1)
    best = group.loc[group["Total Cost ($)"].idxmin()]
    ax.scatter([best["Stock Quantile"]], [best["Total Cost ($)"]], color=colors[label], s=38, zorder=3)
  ax.set_title("Test-set inventory cost by stocking quantile")
  ax.set_xlabel("Stocking quantile")
  ax.set_ylabel("Total simulated cost ($)")
  ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
  ax.grid(True, linestyle=":", alpha=0.35)
  ax.legend(title="Underage to overage cost")
  fig.tight_layout()
  fig.savefig(os.path.join(output_dir, "inventory_cost_by_quantile.png"), dpi=220)
  plt.close(fig)
  return curve


def main():
  df = load_test_predictions()

  c_u, c_o = 3.0, 1.0
  print("=" * 70)
  print(
      f"INVENTORY DECISION SIMULATION (Cu = ${c_u:.2f}, Co = ${c_o:.2f}, Critical"
      f" Ratio = {c_u/(c_u+c_o):.2%})"
  )
  print("=" * 70)

  df_strat = run_strategy_comparison(df, c_u=c_u, c_o=c_o)
  print(
      df_strat.to_string(
          index=False,
          formatters={
              "Total Cost ($)": "${:,.2f}".format,
              "Underage Cost ($)": "${:,.2f}".format,
              "Overage Cost ($)": "${:,.2f}".format,
              "Service Level": "{:.1%}".format,
              "Stockout Rate": "{:.1%}".format,
              "Mean Stock": "{:.2f}".format,
          },
      )
  )

  print("\n" + "=" * 70)
  print("COST SENSITIVITY ACROSS UNDERAGE/OVERAGE RATIOS")
  print("=" * 70)
  df_sens = run_cost_sensitivity(df)
  print(
      df_sens.to_string(
          index=False,
          formatters={
              "Point Cost ($)": "${:,.2f}".format,
              "Quantile Cost ($)": "${:,.2f}".format,
              "Savings (%)": "{:.1f}%".format,
          },
      )
  )

  print("\nCOST CURVE: STOCK QUANTILE VS. TOTAL TEST COST")
  curve = run_stock_quantile_cost_curve(df)
  best = curve.loc[curve.groupby("Cu/Co Ratio")["Total Cost ($)"].idxmin()]
  print(best[["Cu/Co Ratio", "Stock Quantile", "Total Cost ($)"]].to_string(index=False))


if __name__ == "__main__":
  main()
