"""Small read-only API for the saved M5 forecasts and inventory planner."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data/processed/predictions_calibrated.csv"
FEATURES = ROOT / "data/processed/features.csv"
app = FastAPI(title="Demand Forecasting API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])


class InventoryLine(BaseModel):
    store: str
    item: str
    on_hand: float = Field(default=0, ge=0)
    on_order: float = Field(default=0, ge=0)


class OrderRequest(BaseModel):
    date: str
    underage_cost: float = Field(default=3, gt=0)
    overage_cost: float = Field(default=1, gt=0)
    inventory: list[InventoryLine] = Field(default_factory=list)


@lru_cache(maxsize=1)
def predictions():
    if not PREDICTIONS.exists():
        raise HTTPException(503, "Forecast data is missing. Run the pipeline in README.md first.")
    df = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    return df


def _metrics(df):
    y = df.sales.to_numpy(float)
    p = df.pred_q50.to_numpy(float)
    point = df.pred_point.to_numpy(float)
    lo = df.get("cqr_q10", df.pred_q10).to_numpy(float)
    hi = df.get("cqr_q90", df.pred_q90).to_numpy(float)
    raw_lo, raw_hi = df.pred_q10.to_numpy(float), df.pred_q90.to_numpy(float)
    cqr_delta = float(np.mean(hi - raw_hi)) if "cqr_q90" in df else 0.0
    covered = (y >= lo) & (y <= hi)
    positive, zero = y > 0, y == 0
    result = {"rows": int(len(df)), "mae": float(np.mean(abs(y-p))), "rmse": float(np.sqrt(np.mean((y-p)**2))),
            "point_mae": float(np.mean(abs(y-point))), "point_rmse": float(np.sqrt(np.mean((y-point)**2))),
            "coverage": float(np.mean(covered)), "interval_width": float(np.mean(hi-lo)),
            "positive_coverage": float(np.mean(covered[positive])) if positive.any() else None,
            "positive_rows": int(positive.sum()), "zero_coverage": float(np.mean(covered[zero])) if zero.any() else None,
            "zero_rows": int(zero.sum()),
            "raw_coverage": float(np.mean((y >= raw_lo) & (y <= raw_hi))),
            "raw_width": float(np.mean(raw_hi-raw_lo)), "calibration_expansion": cqr_delta,
            "p10_below_strict": float(np.mean(y < df.pred_q10)), "p10_below_or_equal": float(np.mean(y <= df.pred_q10)),
            "p50_below_strict": float(np.mean(y < df.pred_q50)), "p50_below_or_equal": float(np.mean(y <= df.pred_q50)),
            "p90_below_strict": float(np.mean(y < df.pred_q90)), "p90_below_or_equal": float(np.mean(y <= df.pred_q90))}
    for col, prefix in (("baseline_moving_avg", "moving_average"), ("baseline_naive", "weekly_repeat")):
        if col in df:
            baseline = df[col].to_numpy(float)
            result[f"{prefix}_mae"] = float(np.mean(abs(y-baseline)))
            result[f"{prefix}_rmse"] = float(np.sqrt(np.mean((y-baseline)**2)))
    return result


def _interval_summary(df, lower, upper, label):
    y = df.sales.to_numpy(float)
    lo, hi = df[lower].to_numpy(float), df[upper].to_numpy(float)
    return {"method": label, "coverage": float(np.mean((y >= lo) & (y <= hi))), "width": float(np.mean(hi-lo))}


def _inventory_cost(demand, stock, underage, overage):
    return float(np.maximum(0, demand-stock).sum()*underage + np.maximum(0, stock-demand).sum()*overage)


def _quantile(row, tau):
    pairs = []
    for level in (10, 50, 75, 90, 95, 99):
        column = f"pred_q{level:02d}"
        if hasattr(row, column):
            value = getattr(row, column)
            if level == 10 and hasattr(row, "cqr_q10"):
                value = row.cqr_q10
            if level == 90 and hasattr(row, "cqr_q90"):
                value = row.cqr_q90
            pairs.append((level / 100, value))
    if not pairs:
        lower = getattr(row, "cqr_q10", row.pred_q10)
        upper = getattr(row, "cqr_q90", row.pred_q90)
        pairs = [(0.1, lower), (0.5, row.pred_q50), (0.9, upper)]
    return float(max(0, np.interp(tau, [p[0] for p in pairs], [p[1] for p in pairs])))


def _quantile_grid(df, taus):
    levels = []
    columns = []
    for level in (10, 50, 75, 90, 95, 99):
        column = f"pred_q{level:02d}"
        if column in df:
            value = df[column].to_numpy(float)
            if level == 10 and "cqr_q10" in df:
                value = df.cqr_q10.to_numpy(float)
            if level == 90 and "cqr_q90" in df:
                value = df.cqr_q90.to_numpy(float)
            levels.append(level / 100)
            columns.append(value)
    values = np.column_stack(columns)
    result = np.empty((len(df), len(taus)), dtype=float)
    for j, tau in enumerate(taus):
        upper = min(max(int(np.searchsorted(levels, tau, side="right")), 1), len(levels) - 1)
        lower = upper - 1
        weight = (tau - levels[lower]) / (levels[upper] - levels[lower])
        result[:, j] = np.maximum(0.0, values[:, lower] + weight * (values[:, upper] - values[:, lower]))
    return result


@app.get("/api/health")
def health():
    return {"status": "ok", "forecast_data": PREDICTIONS.exists()}


@app.get("/api/options")
def options():
    df = predictions()
    # Put higher-volume products first so the opening view is a meaningful example.
    item_order = df.groupby("item").sales.mean().sort_values(ascending=False).index.tolist()
    return {"stores": sorted(df.store.unique().tolist()), "items": item_order, "dates": sorted(df.date.dt.strftime("%Y-%m-%d").unique().tolist())}


@app.get("/api/dashboard")
def dashboard(store: str, item: str, underage_cost: float = 3, overage_cost: float = 1):
    df = predictions()
    sub = df[(df.store == store) & (df.item == item)].sort_values("date")
    if sub.empty: raise HTTPException(404, "No forecast found for this selection.")
    ratio = underage_cost / (underage_cost + overage_cost)
    rows = []
    for r in sub.itertuples():
        p10, p90 = float(getattr(r, "cqr_q10", r.pred_q10)), float(getattr(r, "cqr_q90", r.pred_q90))
        rows.append({"date": r.date.strftime("%Y-%m-%d"), "actual": float(r.sales), "point": float(r.pred_point), "p10": p10, "p50": float(r.pred_q50), "p90": p90, "interval": max(0.0, p90-p10), "stock": _quantile(r, ratio), "inside_interval": bool(p10 <= float(r.sales) <= p90)})
    # Whole saved test window metrics, plus selected-series context.
    all_metrics = _metrics(df)
    demand = df.sales.to_numpy(float)
    point_stock = df.pred_point.to_numpy(float)
    qstock = np.array([_quantile(r, ratio) for r in df.itertuples()])
    point_cost = _inventory_cost(demand, point_stock, underage_cost, overage_cost)
    quantile_cost = _inventory_cost(demand, qstock, underage_cost, overage_cost)
    strategies = [
        {"strategy": "Point forecast", "cost": point_cost, "stock": point_stock},
        {"strategy": "Median (P50)", "cost": _inventory_cost(demand, df.pred_q50.to_numpy(float), underage_cost, overage_cost), "stock": df.pred_q50.to_numpy(float)},
        {"strategy": f"Cost-based target (P{ratio*100:.0f})", "cost": quantile_cost, "stock": qstock},
        {"strategy": "Conservative (P90)", "cost": _inventory_cost(demand, df.get("cqr_q90", df.pred_q90).to_numpy(float), underage_cost, overage_cost), "stock": df.get("cqr_q90", df.pred_q90).to_numpy(float)},
    ]
    cost_rows = []
    for row in strategies:
        stock = row["stock"]
        missed_cost = float(np.maximum(0, demand-stock).sum()*underage_cost)
        extra_cost = float(np.maximum(0, stock-demand).sum()*overage_cost)
        cost_rows.append({"strategy": row["strategy"], "cost": row["cost"], "missed_cost": missed_cost,
                          "extra_cost": extra_cost, "service_level": float(np.mean(stock >= demand)),
                          "stockout_rate": float(np.mean(demand > stock)), "savings_vs_point": point_cost-row["cost"]})
    interval_methods = [_interval_summary(df, "pred_q10", "pred_q90", "Before calibration")]
    if {"cqr_q10", "cqr_q90"}.issubset(df.columns):
        interval_methods.append(_interval_summary(df, "cqr_q10", "cqr_q90", "After CQR calibration"))
    if {"res_q10", "res_q90"}.issubset(df.columns):
        interval_methods.append(_interval_summary(df, "res_q10", "res_q90", "Residual-based interval (bonus)"))
    volume = df.groupby("item").sales.mean().sort_values()
    products_with_sales = volume[volume > 0]
    if len(products_with_sales) >= 2:
        volume = products_with_sales
    volume_examples = []
    for label, product in (("Lower-volume example", volume.index[0]), ("Higher-volume example", volume.index[-1])):
        part = df[df.item == product]
        m = _metrics(part)
        volume_examples.append({"example": label, "item": product, "mean_daily_sales": float(part.sales.mean()), "mae": m["mae"], "coverage": m["coverage"], "records": len(part)})
    taus = np.round(np.arange(0.10, 1.00, 0.01), 2)
    stock_grid = _quantile_grid(df, taus)
    cost_curve = []
    for index, tau in enumerate(taus):
        stock = stock_grid[:, index]
        total = _inventory_cost(demand, stock, underage_cost, overage_cost)
        cost_curve.append({"quantile": float(tau), "total_cost": total})
    best = min(cost_curve, key=lambda point: point["total_cost"])
    return {"series": rows, "metrics": all_metrics, "series_metrics": _metrics(sub), "interval_methods": interval_methods, "volume_examples": volume_examples, "cost": {"point": point_cost, "quantile": quantile_cost, "savings": point_cost-quantile_cost, "savings_pct": 100*(point_cost-quantile_cost)/point_cost if point_cost else 0, "target_service": ratio, "strategies": cost_rows, "curve": cost_curve, "best_quantile": best["quantile"], "best_cost": best["total_cost"]}}


@app.get("/api/failure-analysis")
def failure_analysis():
    df = predictions().copy()
    if not FEATURES.exists(): return {"groups": [], "note": "Feature file not available; rerun feature generation."}
    cols = ["date", "store", "item"]
    wanted = ["rolling_mean_28", "rolling_std_28", "is_event", "price_ratio"]
    feat = pd.read_csv(FEATURES, usecols=lambda c: c in cols + wanted, parse_dates=["date"])
    df = df.merge(feat.drop_duplicates(cols), on=cols, how="left")
    lo, hi = df.get("cqr_q10", df.pred_q10), df.get("cqr_q90", df.pred_q90)
    df["covered"] = (df.sales >= lo) & (df.sales <= hi)
    masks = {"All test days": pd.Series(True, index=df.index), "Zero sales": df.sales.eq(0), "Positive sales": df.sales.gt(0)}
    if {"rolling_mean_28", "rolling_std_28"}.issubset(df.columns):
        spike = df.sales.gt(df.rolling_mean_28 + 2*df.rolling_std_28).fillna(False)
        masks.update({"Demand spikes": spike, "Usual demand": ~spike})
    if "is_event" in df: masks.update({"Holiday / event": df.is_event.fillna(0).astype(bool), "Regular calendar": ~df.is_event.fillna(0).astype(bool)})
    if "price_ratio" in df:
        promo = df.price_ratio.lt(.95).fillna(False)
        masks.update({"Possible promotion (price-drop proxy)": promo, "No price-drop proxy": ~promo})
    df["interval_width"] = hi-lo
    groups = [{"group": k, "count": int(m.sum()), "coverage": float(df.loc[m, "covered"].mean()) if m.any() else None,
               "mean_width": float(df.loc[m, "interval_width"].mean()) if m.any() else None} for k,m in masks.items()]
    return {"groups": groups, "note": "Price drops are used as a promotion proxy because M5 does not provide complete promotion labels."}


@app.post("/api/order-list")
def order_list(req: OrderRequest):
    df = predictions()
    chosen = df[df.date.eq(pd.Timestamp(req.date))]
    if chosen.empty: raise HTTPException(404, "No forecast data for that date.")
    inv = {(x.store, x.item): x for x in req.inventory}
    tau = req.underage_cost/(req.underage_cost+req.overage_cost)
    result = []
    for r in chosen.itertuples():
        line = inv.get((r.store, r.item))
        on_hand, on_order = (line.on_hand, line.on_order) if line else (0, 0)
        target = _quantile(r, tau)
        result.append({"store": r.store, "item": r.item, "date": req.date, "actual": float(r.sales), "target_stock": target, "on_hand": on_hand, "on_order": on_order, "buy": max(0, int(np.ceil(target-on_hand-on_order)))})
    return {"rows": result, "target_service": tau}
