"""
FastAPI application factory for the semantic router service.

The application exposes:
- lifecycle-managed global `RouteLayer`
- CORS support with configurable origins
- request timing and request ID headers
- structured JSON logging via structlog
- rich OpenAPI documentation at `/docs`
- a health check at `/health`
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import build_route_layer
from api.middleware import configure_middleware
from api.routers import query as query_router
from api.routers import routes as routes_router
from api.schemas import HealthResponse
from semantic_router.layer import RouteLayer


def configure_logging() -> None:
    """
    Configure structlog for JSON-style API logging.

    Algorithm:
        Build a simple processor chain that timestamps and renders structured log
        events through the standard logging system.

    Complexity:
        O(1).

    Example:
        >>> configure_logging.__name__
        'configure_logging'
    """

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _cors_origins_from_env() -> list[str]:
    raw_value = os.getenv("SEMANTIC_ROUTER_CORS_ORIGINS", "*")
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


def create_app(
    route_layer: RouteLayer | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """
    Create and configure the semantic router FastAPI application.

    Algorithm:
        Configure structured logging, create a lifespan-managed route layer,
        attach middleware, and register routers and health endpoints.

    Complexity:
        O(1) during startup before routes are added.

    Args:
        route_layer: Optional prebuilt route layer, mainly for tests.
        cors_origins: Optional explicit CORS origin allowlist.

    Returns:
        Configured FastAPI application.

    Example:
        >>> isinstance(create_app(), FastAPI)
        True
    """

    configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.route_layer = route_layer or build_route_layer()
        yield

    app = FastAPI(
        title="Semantic Router API",
        description=(
            "Route user queries to handlers using vector similarity. "
            "This API exposes route CRUD, routing, embedding, calibration helpers, "
            "and save/load operations."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or _cors_origins_from_env(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    configure_middleware(app)

    app.include_router(routes_router.router)
    app.include_router(query_router.router)

    @app.get("/health", response_model=HealthResponse, tags=["health"], summary="Health check")
    async def health() -> HealthResponse:
        layer = cast(RouteLayer, app.state.route_layer)
        return HealthResponse(status="ok", routes_loaded=len(layer.list_routes()))

    return app


app = create_app()


def run() -> None:
    """
    Launch the development API server with Uvicorn.

    Algorithm:
        Delegate directly to `uvicorn.run` with the module-level application.

    Complexity:
        O(1).

    Example:
        >>> run.__name__
        'run'
    """

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
