"""Inventory cost and quantile based replenishment calculations."""

import numpy as np


def critical_ratio(underage_cost, overage_cost):
    """Return the Newsvendor ratio Cu / (Cu + Co).

    Cu is the loss from one unit too few and Co the loss from one unit too
    many. Under linear single-period costs, this ratio is the demand quantile
    that minimizes expected inventory cost.
    """
    if underage_cost <= 0 or overage_cost <= 0:
        raise ValueError("underage_cost and overage_cost must be positive")
    return float(underage_cost / (underage_cost + overage_cost))


def inventory_cost(demand, stock, underage_cost, overage_cost):
    return float(
        np.maximum(0, demand - stock).sum() * underage_cost
        + np.maximum(0, stock - demand).sum() * overage_cost
    )


def row_quantile(row, tau):
    """Interpolate available forecast quantiles at probability ``tau``."""
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
    pairs.sort(key=lambda pair: pair[0])
    return float(max(0, np.interp(tau, [p[0] for p in pairs], [p[1] for p in pairs])))


def recommended_safety_stock(row, underage_cost, overage_cost):
    """Estimate Newsvendor stock at CR = Cu / (Cu + Co), interpolating quantiles."""
    return row_quantile(row, critical_ratio(underage_cost, overage_cost))


def quantile_grid(df, taus):
    levels, columns = [], []
    for level in (10, 50, 75, 90, 95, 99):
        column = f"pred_q{level:02d}"
        if column in df:
            values = df[column].to_numpy(float)
            if level == 10 and "cqr_q10" in df:
                values = df.cqr_q10.to_numpy(float)
            if level == 90 and "cqr_q90" in df:
                values = df.cqr_q90.to_numpy(float)
            levels.append(level / 100)
            columns.append(values)
    values = np.column_stack(columns)
    result = np.empty((len(df), len(taus)), dtype=float)
    for index, tau in enumerate(taus):
        upper = min(
            max(int(np.searchsorted(levels, tau, side="right")), 1), len(levels) - 1
        )
        lower = upper - 1
        weight = (tau - levels[lower]) / (levels[upper] - levels[lower])
        result[:, index] = np.maximum(
            0.0, values[:, lower] + weight * (values[:, upper] - values[:, lower])
        )
    return result
