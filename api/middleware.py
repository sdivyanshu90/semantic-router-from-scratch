"""Request middleware, logging, and metrics for the semantic router API."""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = structlog.get_logger(__name__)

ROUTING_REQUESTS_TOTAL = Counter(
    "routing_requests_total",
    "Total routing decisions processed by the API.",
    labelnames=("matched_route",),
)
ROUTING_LATENCY_SECONDS = Histogram(
    "routing_latency_seconds",
    "Latency for routing API requests.",
)
ROUTING_NO_MATCH_TOTAL = Counter(
    "routing_no_match_total",
    "Total routing decisions that produced no match.",
)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Add request IDs, timing headers, structured logs, and routing metrics.

    Algorithm:
        Generate or propagate a request ID, measure total request latency, attach
        headers to the response, then emit one structured log event per routing
        result captured by endpoint handlers.

    Complexity:
        O(E) where E is the number of routing events recorded on the request.

    Example:
        >>> RequestLoggingMiddleware.__name__
        'RequestLoggingMiddleware'
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        started_at = time.perf_counter()
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        elapsed_seconds = time.perf_counter() - started_at
        latency_ms = round(elapsed_seconds * 1000.0, 3)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(elapsed_seconds)

        routing_events = getattr(request.state, "routing_events", [])
        if routing_events:
            ROUTING_LATENCY_SECONDS.observe(elapsed_seconds)
            for event in routing_events:
                matched_route = event.get("matched_route") or "none"
                ROUTING_REQUESTS_TOTAL.labels(matched_route=matched_route).inc()
                if matched_route == "none":
                    ROUTING_NO_MATCH_TOTAL.inc()
                logger.info(
                    "routing_request",
                    request_id=request_id,
                    route=request.url.path,
                    query_length=event.get("query_length", 0),
                    matched_route=event.get("matched_route"),
                    score=event.get("score"),
                    latency_ms=latency_ms,
                )
        else:
            logger.info(
                "http_request",
                request_id=request_id,
                route=request.url.path,
                status_code=response.status_code,
                latency_ms=latency_ms,
            )
        return response


def configure_middleware(app: FastAPI) -> None:
    """
    Attach middleware and Prometheus instrumentation to a FastAPI app.

    Algorithm:
        Add the custom request logging middleware and expose `/metrics` through
        the Prometheus FastAPI instrumentator.

    Complexity:
        O(1).

    Args:
        app: FastAPI application instance.

    Example:
        >>> configure_middleware.__name__
        'configure_middleware'
    """

    app.add_middleware(RequestLoggingMiddleware)
    Instrumentator().instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")
