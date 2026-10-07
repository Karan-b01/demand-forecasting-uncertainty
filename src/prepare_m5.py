"""Convert the official M5 wide sales file into the project's daily panel."""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


def prepare_m5(
    raw_dir: str = "data/raw",
    output: str = "data/processed/clean_sales.csv",
    item_count: int = 100,
    selection: str = "demand-spread",
) -> Path:
    raw = Path(raw_dir)
    sales_path = raw / "sales_train_evaluation.csv"
    calendar_path = raw / "calendar.csv"
    prices_path = raw / "sell_prices.csv"
    for path in (sales_path, calendar_path, prices_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download the M5 Forecasting data from Kaggle "
                "and place sales_train_evaluation.csv, calendar.csv, and "
                "sell_prices.csv in data/raw/."
            )

    sales = pd.read_csv(sales_path)
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_columns = [name for name in sales.columns if name.startswith("d_")]

    # Select product IDs spread over the full demand range, retaining each
    # selected product in every store. This keeps the project runnable while
    # making the evaluation broader than a single store or low-volume sample.
    item_average = sales.groupby("item_id", sort=True)[day_columns].mean().mean(axis=1)
    ranked_items = item_average.sort_values().index.to_numpy()
    selected_count = min(max(1, item_count), len(ranked_items))
    if selection == "high-volume":
        selected_items = ranked_items[-selected_count:]
    elif selection == "demand-spread":
        selected_positions = np.linspace(
            0, len(ranked_items) - 1, selected_count
        ).round().astype(int)
        selected_items = ranked_items[selected_positions]
    else:
        raise ValueError("selection must be 'demand-spread' or 'high-volume'")
    sales = sales[sales["item_id"].isin(selected_items)].copy()
    print(
        f"Selected {selected_count} product IDs across "
        f"{sales['store_id'].nunique()} stores using '{selection}' selection."
    )

    long = sales.melt(
        id_vars=id_columns, value_vars=day_columns,
        var_name="d", value_name="sales",
    )
    long = long.rename(columns={"store_id": "store", "item_id": "item"})

    calendar = pd.read_csv(calendar_path)
    keep_calendar = [
        col for col in ["d", "date", "wm_yr_wk", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]
        if col in calendar.columns
    ]
    long = long.merge(calendar[keep_calendar], on="d", how="left", validate="many_to_one")

    prices = pd.read_csv(prices_path)
    long = long.merge(
        prices.rename(columns={"store_id": "store", "item_id": "item"}),
        on=["store", "item", "wm_yr_wk"], how="left", validate="many_to_one",
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(output_path, index=False)
    print(f"Wrote {len(long):,} daily SKU-store rows to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--item-count", type=int, default=100,
        help="Number of distinct products to keep across every store (default: 100).",
    )
    parser.add_argument(
        "--selection", choices=["demand-spread", "high-volume"],
        default="demand-spread",
        help="Sample across demand levels, or keep the highest-selling products.",
    )
    args = parser.parse_args()
    prepare_m5(item_count=args.item_count, selection=args.selection)
