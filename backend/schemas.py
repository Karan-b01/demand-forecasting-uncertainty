"""Validated request models used by the HTTP API."""

from datetime import date

from pydantic import BaseModel, Field


class InventoryLine(BaseModel):
    store: str
    item: str
    on_hand: float = Field(default=0, ge=0)
    on_order: float = Field(default=0, ge=0)


class OrderRequest(BaseModel):
    date: date
    underage_cost: float = Field(default=3, gt=0)
    overage_cost: float = Field(default=1, gt=0)
    inventory: list[InventoryLine] = Field(default_factory=list)


class SimulationRequest(BaseModel):
    """What-if edits applied to a historical feature row before inference."""

    store_id: str
    item_id: str
    date: date
    price_change_pct: float = Field(default=0, ge=-50, le=50)
    is_holiday_override: bool = False
