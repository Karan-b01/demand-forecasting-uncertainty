"""FastAPI application entry point for the demand forecasting API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.core.config import API_TITLE, API_VERSION, CORS_ORIGINS
from backend.services.inference import load_models


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load fitted inference models once when the API process starts."""
    try:
        application.state.inference_models = load_models()
        application.state.inference_error = None
    except FileNotFoundError as exc:
        application.state.inference_models = None
        application.state.inference_error = str(exc)
    yield


def create_app() -> FastAPI:
    """Build and configure the HTTP application."""
    application = FastAPI(title=API_TITLE, version=API_VERSION, lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()
