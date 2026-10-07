import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
  """Extracts temporal calendar signals."""
  df = df.copy()
  df['date'] = pd.to_datetime(df['date'])

  df['dayofweek'] = df['date'].dt.dayofweek
  df['day'] = df['date'].dt.day
  df['month'] = df['date'].dt.month
  df['year'] = df['date'].dt.year
  df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
  df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
  df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

  # Fill missing event names as 'None' and create binary holiday flag
  if 'event_name_1' in df.columns:
    df['event_name_1'] = df['event_name_1'].fillna('None').astype(str)
    df['is_event'] = (df['event_name_1'] != 'None').astype(int)
  else:
    df['is_event'] = 0

  return df


def add_lag_and_rolling_features(
    df: pd.DataFrame, horizon: int = 7
) -> pd.DataFrame:
  """Calculates lags and rolling stats anchored strictly at t - horizon.

  This guarantees zero leakage into any t + horizon forecast.
  """
  df = df.copy()
  df['date'] = pd.to_datetime(df['date'])
  df = df.sort_values(['store', 'item', 'date']).reset_index(drop=True)

  group = df.groupby(['store', 'item'])['sales']

  # 1. Direct Lags (must be >= horizon)
  lags = [horizon, horizon + 7, horizon + 14, horizon + 21, horizon + 28]
  for lag in lags:
    df[f'sales_lag_{lag}'] = group.shift(lag)

  # 2. Shifted Rolling Statistics
  # Shifting by horizon ensures window looks only at historical data
  windows = [7, 14, 28]
  for w in windows:
    shifted_roll = group.shift(horizon).rolling(window=w, min_periods=w)
    df[f'rolling_mean_{w}'] = shifted_roll.mean()
    df[f'rolling_std_{w}'] = shifted_roll.std()
    df[f'rolling_min_{w}'] = shifted_roll.min()
    df[f'rolling_max_{w}'] = shifted_roll.max()

  # 3. Price change / promo ratio
  if 'sell_price' in df.columns:
    price_group = df.groupby(['store', 'item'])['sell_price']
    df['price_lag_7'] = price_group.shift(horizon)
    df['price_rolling_mean_28'] = (
        price_group.shift(horizon).rolling(28, min_periods=7).mean()
    )
    df['price_ratio'] = df['price_lag_7'] / (df['price_rolling_mean_28'] + 1e-5)

  return df


def generate_feature_dataset(
    input_csv_path: str = 'data/processed/clean_sales.csv',
    horizon: int = 7,
    warmup_lags: int = 35,
) -> pd.DataFrame:
  """Loads clean sales, creates features, drops warmup NaNs, and saves."""
  print(f'Loading data from {input_csv_path}...')
  df = pd.read_csv(input_csv_path)

  print('Generating calendar features...')
  df = add_calendar_features(df)

  print(
      f'Generating lag and rolling features (forecast horizon = {horizon}d)...'
  )
  df = add_lag_and_rolling_features(df, horizon=horizon)

  # Drop warmup rows (first ~35 days contain NaNs due to rolling/lag lookbacks)
  initial_len = len(df)
  df_features = df.dropna().reset_index(drop=True)
  print(
      f'Features created: {len(df_features):,} rows (dropped'
      f' {initial_len - len(df_features):,} warmup rows).'
  )

  return df_features


if __name__ == '__main__':
  features_df = generate_feature_dataset()
  out_csv = 'data/processed/features.csv'
  features_df.to_csv(out_csv, index=False)
  print(f'Successfully saved final feature dataset to {out_csv}')