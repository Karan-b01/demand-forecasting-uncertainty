import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts temporal calendar signals."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["dayofweek"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # Events are sparse by design; encode missing names, rather than dropping
    # almost every ordinary calendar day during feature cleanup.
    for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        if col in df.columns:
            df[col] = df[col].fillna("None").astype(str)
    if "event_name_1" in df.columns or "event_name_2" in df.columns:
        events = [
            df[col].ne("None")
            for col in ["event_name_1", "event_name_2"]
            if col in df.columns
        ]
        df["is_event"] = np.logical_or.reduce(events).astype(int)
    else:
        df["is_event"] = 0

    return df


def add_lag_and_rolling_features(
    df: pd.DataFrame, lead_time: int = 1, *, horizon: int | None = None
) -> pd.DataFrame:
    """Create history-only features for a forecast with ``lead_time`` days' notice.

    The most recent target value used for a forecast dated *t* is from
    ``t - lead_time``. Rolling windows therefore end at that same day. ``horizon``
    remains as a keyword alias for callers of the original API.
    """
    if horizon is not None:
        lead_time = horizon
    if not isinstance(lead_time, int) or lead_time < 1:
        raise ValueError("lead_time must be a positive integer")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)

    group = df.groupby(["store", "item"], sort=False)["sales"]

    # Preserve the lag-7 feature used by the batch seasonal-naive baseline,
    # while anchoring the model's direct lag sequence to the requested lead.
    lags = sorted(
        set(
            [
                7,
                lead_time,
                lead_time + 7,
                lead_time + 14,
                lead_time + 21,
                lead_time + 28,
            ]
        )
    )
    for lag in lags:
        df[f"sales_lag_{lag}"] = group.shift(lag)

    # 2. Shifted Rolling Statistics
    # Shifting by lead_time keeps every rolling window in known history.
    windows = [7, 14, 28]
    for w in windows:
        shifted = group.shift(lead_time)
        rolling_group = shifted.groupby([df["store"], df["item"]], sort=False)
        df[f"rolling_mean_{w}"] = rolling_group.transform(
            lambda values: values.rolling(window=w, min_periods=w).mean()
        )
        df[f"rolling_std_{w}"] = rolling_group.transform(
            lambda values: values.rolling(window=w, min_periods=w).std()
        )
        df[f"rolling_min_{w}"] = rolling_group.transform(
            lambda values: values.rolling(window=w, min_periods=w).min()
        )
        df[f"rolling_max_{w}"] = rolling_group.transform(
            lambda values: values.rolling(window=w, min_periods=w).max()
        )

    # Lower-tail history is especially useful for P10 models on intermittent
    # series: it distinguishes high-volume items' ordinary low days from the
    # many genuinely near-zero items in the pooled training set.
    for w in [28, 56]:
        df[f"rolling_q10_{w}"] = rolling_group.transform(
            lambda values: values.rolling(window=w, min_periods=w).quantile(0.10)
        )

    # 3. Price change / promo ratio
    if "sell_price" in df.columns:
        price_group = df.groupby(["store", "item"])["sell_price"]
        df[f"price_lag_{lead_time}"] = price_group.shift(lead_time)
        shifted_price = price_group.shift(lead_time)
        price_rolling_group = shifted_price.groupby(
            [df["store"], df["item"]], sort=False
        )
        df["price_rolling_mean_28"] = price_rolling_group.transform(
            lambda values: values.rolling(28, min_periods=7).mean()
        )
        df["price_ratio"] = df[f"price_lag_{lead_time}"] / (
            df["price_rolling_mean_28"] + 1e-5
        )

    return df


def generate_feature_dataset(
    input_csv_path: str = "data/processed/clean_sales.csv",
    lead_time: int = 1,
    warmup_lags: int = 35,
    *,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Loads clean sales, creates features, drops warmup NaNs, and saves."""
    if horizon is not None:
        lead_time = horizon
    print(f"Loading data from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)

    print("Generating calendar features...")
    df = add_calendar_features(df)

    print(f"Generating lag and rolling features (forecast lead time = {lead_time}d)...")
    df = add_lag_and_rolling_features(df, lead_time=lead_time)

    # Remove only rows without the sales history required by the model. Prices
    # can legitimately be missing before an item is sold in a store; LightGBM
    # handles those missing optional price signals natively.
    required_history = [
        f"sales_lag_{lag}"
        for lag in sorted(
            set(
                [
                    7,
                    lead_time,
                    lead_time + 7,
                    lead_time + 14,
                    lead_time + 21,
                    lead_time + 28,
                ]
            )
        )
    ] + [f"rolling_mean_{window}" for window in [7, 14, 28]]
    required_history += [f"rolling_q10_{window}" for window in [28, 56]]
    initial_len = len(df)
    df_features = df.dropna(subset=required_history).reset_index(drop=True)
    print(
        f"Features created: {len(df_features):,} rows (dropped"
        f" {initial_len - len(df_features):,} warmup rows)."
    )

    return df_features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate leakage-safe forecast features."
    )
    parser.add_argument(
        "--lead-time",
        type=int,
        default=1,
        help="Forecast lead time in days (default: next-day, 1).",
    )
    args = parser.parse_args()
    features_df = generate_feature_dataset(lead_time=args.lead_time)
    out_csv = "data/processed/features.csv"
    features_df.to_csv(out_csv, index=False)
    print(f"Successfully saved final feature dataset to {out_csv}")
