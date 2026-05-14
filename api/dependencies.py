"""Shared dependencies for the semantic router FastAPI application."""

from __future__ import annotations

from fastapi import Request

from semantic_router.cache import EmbeddingCache
from semantic_router.config import RouterConfig
from semantic_router.encoders import SentenceTransformerEncoder
from semantic_router.layer import RouteLayer


def build_route_layer() -> RouteLayer:
    """
    Construct the default application route layer.

    Algorithm:
        Create a local sentence-transformers encoder backed by the shared embedding
        cache and initialize an empty `RouteLayer` around it.

    Complexity:
        O(1) until routes are added.

    Returns:
        Ready-to-use `RouteLayer` singleton.

    Example:
        >>> isinstance(build_route_layer(), RouteLayer)
        True
    """

    cache = EmbeddingCache()
    encoder = SentenceTransformerEncoder(show_progress=False, cache=cache)
    return RouteLayer(routes=[], encoder=encoder, config=RouterConfig())


def get_route_layer(request: Request) -> RouteLayer:
    """
    Retrieve the application-scoped `RouteLayer` instance.

    Algorithm:
        Read the singleton from `request.app.state`, where the lifespan manager
        stores it during application startup.

    Complexity:
        O(1).

    Args:
        request: FastAPI request object.

    Returns:
        Shared route layer instance.

    Example:
        >>> get_route_layer.__name__
        'get_route_layer'
    """

    return request.app.state.route_layer  # type: ignore[no-any-return]
