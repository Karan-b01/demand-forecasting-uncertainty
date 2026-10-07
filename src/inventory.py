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


def run_strategy_comparison(
    df: pd.DataFrame, c_u: float = 3.0, c_o: float = 1.0
) -> pd.DataFrame:
  """Compares stocking policies against test actuals."""
  demand = df["sales"].values

  # Strategy 1: Stocking at Point Forecast (RMSE model)
  stock_point = df["pred_point"].values

  # Strategy 2: Stocking at Median (q50)
  stock_median = df["pred_q50"].values

  # Strategy 3: Classical Safety Stock (Mean + 0.674 * Std for 75% normal service level)
  # Standard deviation estimated from test residuals
  sigma_est = np.std(demand - stock_point)
  z_75 = 0.674  # Normal inverse CDF for 0.75
  stock_safety = np.maximum(0.0, stock_point + z_75 * sigma_est)

  # Strategy 4: Quantile Newsvendor Policy
  # Critical ratio tau* = 3 / (3 + 1) = 0.75
  # Linear interpolation between q50 and q90
  # tau* is 62.5% of the distance from 0.50 to 0.90
  weight = (0.75 - 0.50) / (0.90 - 0.50)
  stock_quantile_75 = df["pred_q50"].values + weight * (
      df["pred_q90"].values - df["pred_q50"].values
  )

  # Strategy 5: Conservative Upper Band (q90)
  stock_q90 = df["pred_q90"].values

  strategies = {
      "1. Point Forecast (RMSE)": stock_point,
      "2. Median Forecast (q50)": stock_median,
      "3. Point + Gaussian Safety Stock": stock_safety,
      "4. Optimal Quantile Policy (tau*=0.75)": stock_quantile_75,
      "5. Conservative P90": stock_q90,
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

    # Calculate optimal stock via quantile interpolation
    if tau_star <= 0.50:
      stock_opt = df["pred_q10"].values + (tau_star / 0.50) * (
          df["pred_q50"].values - df["pred_q10"].values
      )
    else:
      w = (tau_star - 0.50) / (0.90 - 0.50)
      stock_opt = df["pred_q50"].values + w * (
          df["pred_q90"].values - df["pred_q50"].values
      )

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


if __name__ == "__main__":
  main()