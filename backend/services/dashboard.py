"""Dashboard response assembly from forecast data and business metrics."""

import numpy as np
import pandas as pd
from fastapi import HTTPException

from backend.services.inventory import inventory_cost, quantile_grid, row_quantile
from backend.services.metrics import forecast_metrics, interval_summary


def dashboard_data(df, store, item, underage_cost=3, overage_cost=1):
    series = df[(df.store == store) & (df.item == item)].sort_values("date")
    if series.empty:
        raise HTTPException(404, "No forecast found for this selection.")
    ratio = underage_cost / (underage_cost + overage_cost)
    rows = []
    for row in series.itertuples():
        p10 = float(getattr(row, "cqr_q10", row.pred_q10))
        p90 = float(getattr(row, "cqr_q90", row.pred_q90))
        rows.append(
            {
                "date": row.date.strftime("%Y-%m-%d"),
                "actual": float(row.sales),
                "point": float(row.pred_point),
                "p10": p10,
                "p50": float(row.pred_q50),
                "p90": p90,
                "interval": max(0.0, p90 - p10),
                "stock": row_quantile(row, ratio),
                "inside_interval": bool(p10 <= float(row.sales) <= p90),
            }
        )

    demand = df.sales.to_numpy(float)
    point_stock = df.pred_point.to_numpy(float)
    qstock = np.array([row_quantile(row, ratio) for row in df.itertuples()])
    point_cost = inventory_cost(demand, point_stock, underage_cost, overage_cost)
    quantile_cost = inventory_cost(demand, qstock, underage_cost, overage_cost)
    strategies = [
        ("Point forecast", point_stock, point_cost),
        (
            "Median (P50)",
            df.pred_q50.to_numpy(float),
            inventory_cost(
                demand, df.pred_q50.to_numpy(float), underage_cost, overage_cost
            ),
        ),
        (f"Cost-based target (P{ratio*100:.0f})", qstock, quantile_cost),
        (
            "Conservative (P90)",
            df.get("cqr_q90", df.pred_q90).to_numpy(float),
            inventory_cost(
                demand,
                df.get("cqr_q90", df.pred_q90).to_numpy(float),
                underage_cost,
                overage_cost,
            ),
        ),
    ]
    cost_rows = []
    for label, stock, cost in strategies:
        cost_rows.append(
            {
                "strategy": label,
                "cost": cost,
                "missed_cost": float(
                    np.maximum(0, demand - stock).sum() * underage_cost
                ),
                "extra_cost": float(np.maximum(0, stock - demand).sum() * overage_cost),
                "service_level": float(np.mean(stock >= demand)),
                "stockout_rate": float(np.mean(demand > stock)),
                "savings_vs_point": point_cost - cost,
            }
        )

    intervals = [interval_summary(df, "pred_q10", "pred_q90", "Before calibration")]
    if {"cqr_q10", "cqr_q90"}.issubset(df.columns):
        intervals.append(
            interval_summary(df, "cqr_q10", "cqr_q90", "After CQR calibration")
        )
    if {"res_q10", "res_q90"}.issubset(df.columns):
        intervals.append(
            interval_summary(
                df, "res_q10", "res_q90", "Residual-based interval (bonus)"
            )
        )

    volume = df.groupby("item").sales.mean().sort_values()
    positive_volume = volume[volume > 0]
    if len(positive_volume) >= 2:
        volume = positive_volume
    examples = []
    for label, product in (
        ("Lower-volume example", volume.index[0]),
        ("Higher-volume example", volume.index[-1]),
    ):
        part = df[df.item == product]
        metrics = forecast_metrics(part)
        examples.append(
            {
                "example": label,
                "item": product,
                "mean_daily_sales": float(part.sales.mean()),
                "mae": metrics["mae"],
                "coverage": metrics["coverage"],
                "records": len(part),
            }
        )

    taus = np.round(np.arange(0.10, 1.00, 0.01), 2)
    stock_grid = quantile_grid(df, taus)
    curve = [
        {
            "quantile": float(tau),
            "total_cost": inventory_cost(
                demand, stock_grid[:, i], underage_cost, overage_cost
            ),
        }
        for i, tau in enumerate(taus)
    ]
    best = min(curve, key=lambda point: point["total_cost"])
    return {
        "series": rows,
        "metrics": forecast_metrics(df),
        "series_metrics": forecast_metrics(series),
        "interval_methods": intervals,
        "volume_examples": examples,
        "cost": {
            "point": point_cost,
            "quantile": quantile_cost,
            "savings": point_cost - quantile_cost,
            "savings_pct": (
                100 * (point_cost - quantile_cost) / point_cost if point_cost else 0
            ),
            "target_service": ratio,
            "strategies": cost_rows,
            "curve": curve,
            "best_quantile": best["quantile"],
            "best_cost": best["total_cost"],
        },
    }
