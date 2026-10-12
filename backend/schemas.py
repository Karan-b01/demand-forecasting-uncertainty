"""Validated request models used by the HTTP API."""

from pydantic import BaseModel, Field


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
