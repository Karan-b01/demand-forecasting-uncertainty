"""Purchase order recommendations from forecast quantiles and inventory."""

import numpy as np
import pandas as pd
from fastapi import HTTPException

from backend.schemas import OrderRequest
from backend.services.inventory import critical_ratio, recommended_safety_stock


def build_order_list(df, request: OrderRequest):
    chosen = df[df.date.eq(pd.Timestamp(request.date))]
    if chosen.empty:
        raise HTTPException(404, "No forecast data for that date.")
    inventory = {(line.store, line.item): line for line in request.inventory}
    tau = critical_ratio(request.underage_cost, request.overage_cost)
    rows = []
    for row in chosen.itertuples():
        line = inventory.get((row.store, row.item))
        on_hand, on_order = (line.on_hand, line.on_order) if line else (0, 0)
        target = recommended_safety_stock(
            row, request.underage_cost, request.overage_cost
        )
        rows.append(
            {
                "store": row.store,
                "item": row.item,
                "date": request.date,
                "actual": float(row.sales),
                "target_stock": target,
                "recommended_safety_stock": target,
                "on_hand": on_hand,
                "on_order": on_order,
                "buy": max(0, int(np.ceil(target - on_hand - on_order))),
            }
        )
    return {"rows": rows, "target_service": tau}
