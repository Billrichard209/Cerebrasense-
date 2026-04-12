"""FastAPI application entry point for the structural MRI backend core."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.storage import connect_backend_storage

from .routers import explainability, governance, inference, longitudinal, system, volumetrics


@asynccontextmanager
async def _app_lifespan(_: FastAPI):
    """Initialize lightweight backend storage when the API starts."""

    connection = connect_backend_storage()
    connection.close()
    yield


def create_app() -> FastAPI:
    """Create the FastAPI application with modular routers."""

    application = FastAPI(
        title="Structural MRI Backend Core",
        version="0.3.0",
        lifespan=_app_lifespan,
        description=(
            "OASIS-first structural MRI decision-support backend for inference, "
            "explainability, volumetrics, and longitudinal workflows."
        ),
    )
    application.include_router(system.router)
    application.include_router(governance.router)
    application.include_router(volumetrics.router)
    application.include_router(inference.router)
    application.include_router(explainability.router)
    application.include_router(longitudinal.router)
    return application


app = create_app()
