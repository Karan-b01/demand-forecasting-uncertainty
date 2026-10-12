"""FastAPI application entry point for the demand forecasting API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.core.config import API_TITLE, API_VERSION, CORS_ORIGINS


def create_app() -> FastAPI:
    """Build and configure the HTTP application."""
    application = FastAPI(title=API_TITLE, version=API_VERSION)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()
