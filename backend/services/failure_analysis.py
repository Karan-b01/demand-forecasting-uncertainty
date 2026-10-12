"""Coverage breakdowns for forecast failure analysis."""

import pandas as pd

from backend.core.config import FEATURES_PATH


def analyze_failures(predictions):
    df = predictions.copy()
    if not FEATURES_PATH.exists():
        return {
            "groups": [],
            "note": "Feature file not available; rerun feature generation.",
        }
    keys = ["date", "store", "item"]
    wanted = ["rolling_mean_28", "rolling_std_28", "is_event", "price_ratio"]
    features = pd.read_csv(
        FEATURES_PATH, usecols=lambda col: col in keys + wanted, parse_dates=["date"]
    )
    df = df.merge(features.drop_duplicates(keys), on=keys, how="left")
    lower, upper = df.get("cqr_q10", df.pred_q10), df.get("cqr_q90", df.pred_q90)
    df["covered"] = (df.sales >= lower) & (df.sales <= upper)
    masks = {
        "All test days": pd.Series(True, index=df.index),
        "Zero sales": df.sales.eq(0),
        "Positive sales": df.sales.gt(0),
    }
    if {"rolling_mean_28", "rolling_std_28"}.issubset(df.columns):
        spike = df.sales.gt(df.rolling_mean_28 + 2 * df.rolling_std_28).fillna(False)
        masks.update({"Demand spikes": spike, "Usual demand": ~spike})
    if "is_event" in df:
        event = df.is_event.fillna(0).astype(bool)
        masks.update({"Holiday / event": event, "Regular calendar": ~event})
    if "price_ratio" in df:
        promo = df.price_ratio.lt(0.95).fillna(False)
        masks.update(
            {
                "Possible promotion (price-drop proxy)": promo,
                "No price-drop proxy": ~promo,
            }
        )
    df["interval_width"] = upper - lower
    groups = [
        {
            "group": label,
            "count": int(mask.sum()),
            "coverage": float(df.loc[mask, "covered"].mean()) if mask.any() else None,
            "mean_width": (
                float(df.loc[mask, "interval_width"].mean()) if mask.any() else None
            ),
        }
        for label, mask in masks.items()
    ]
    return {
        "groups": groups,
        "note": "Price drops are used as a promotion proxy because M5 does not provide complete promotion labels.",
    }
