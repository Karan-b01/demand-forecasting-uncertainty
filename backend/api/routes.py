"""Versioned API endpoints; business calculations live in services."""

from fastapi import APIRouter, HTTPException, Query, Request

from backend.core.config import PREDICTIONS_PATH
from backend.schemas import OrderRequest, SimulationRequest
from backend.services.dashboard import dashboard_data
from backend.services.failure_analysis import analyze_failures
from backend.services.forecast_data import load_predictions
from backend.services.inference import simulate_forecast
from backend.services.orders import build_order_list

router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    return {"status": "ok", "forecast_data": PREDICTIONS_PATH.exists()}


@router.get("/options")
def options():
    df = load_predictions()
    item_order = (
        df.groupby("item").sales.mean().sort_values(ascending=False).index.tolist()
    )
    return {
        "stores": sorted(df.store.unique().tolist()),
        "items": item_order,
        "dates": sorted(df.date.dt.strftime("%Y-%m-%d").unique().tolist()),
    }


@router.get("/dashboard")
def dashboard(
    store: str,
    item: str,
    underage_cost: float = Query(default=3, gt=0),
    overage_cost: float = Query(default=1, gt=0),
):
    return dashboard_data(load_predictions(), store, item, underage_cost, overage_cost)


@router.get("/failure-analysis")
def failure_analysis():
    return analyze_failures(load_predictions())


@router.post("/order-list")
def order_list(request: OrderRequest):
    return build_order_list(load_predictions(), request)


@router.post("/v1/forecast/simulate")
def forecast_simulation(payload: SimulationRequest, request: Request):
    """Run in-memory what-if inference against a stored feature row."""
    models = getattr(request.app.state, "inference_models", None)
    if models is None:
        detail = getattr(
            request.app.state, "inference_error", "Models are unavailable."
        )
        raise HTTPException(status_code=503, detail=detail)
    try:
        return simulate_forecast(payload, models=models)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=500, detail=f"Inference configuration error: {exc}"
        ) from exc
