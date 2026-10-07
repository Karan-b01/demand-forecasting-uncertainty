import numpy as np
import pandas as pd
from src.features import add_lag_and_rolling_features


def test_no_leakage_within_horizon():
  """Asserts that mutating sales data inside the horizon window [t - horizon + 1, t]

  does NOT alter the feature row generated for date t.
  """
  dates = pd.date_range('2015-01-01', periods=80, freq='D')
  df_base = pd.DataFrame({
      'date': dates,
      'store': 'CA_1',
      'item': 'FOODS_3_001',
      'sales': np.random.poisson(lam=20, size=80).astype(float),
      'sell_price': 2.50,
  })

  horizon = 7
  df_clean_feat = add_lag_and_rolling_features(df_base.copy(), horizon=horizon)

  # Pick evaluation cutoff index (day 45)
  cutoff_idx = 45
  cutoff_date = dates[cutoff_idx]

  # Tamper with the sales strictly inside the protected horizon window:
  # dates between [t - 6] and [t]
  df_tampered = df_base.copy()
  tamper_mask = (df_tampered['date'] > dates[cutoff_idx - horizon]) & (
      df_tampered['date'] <= cutoff_date
  )
  df_tampered.loc[tamper_mask, 'sales'] += 999.0  # Inject massive spike

  df_tampered_feat = add_lag_and_rolling_features(
      df_tampered.copy(), horizon=horizon
  )

  # Extract feature vector for cutoff_date (excluding target sales)
  feat_cols = [
      c
      for c in df_clean_feat.columns
      if c not in ['sales', 'date', 'store', 'item']
  ]
  row_orig = df_clean_feat[df_clean_feat['date'] == cutoff_date][
      feat_cols
  ].reset_index(drop=True)
  row_tampered = df_tampered_feat[df_tampered_feat['date'] == cutoff_date][
      feat_cols
  ].reset_index(drop=True)

  # Check that features generated for date t are identical
  pd.testing.assert_frame_equal(row_orig, row_tampered)