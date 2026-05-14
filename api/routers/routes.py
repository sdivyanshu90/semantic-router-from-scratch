"""CRUD endpoints for managing semantic routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_route_layer
from api.schemas import (
    OperationStatusResponse,
    RouteCreateRequest,
    RouteDetailResponse,
    RouteSummaryResponse,
    RouteTestRequest,
    RouteTestResponse,
    RouteUpdateRequest,
)
from semantic_router.layer import RouteLayer
from semantic_router.route import Route, _load_handler

router = APIRouter(prefix="/routes", tags=["routes"])
LayerDependency = Annotated[RouteLayer, Depends(get_route_layer)]


def _to_detail(route: Route) -> RouteDetailResponse:
    return RouteDetailResponse(
        name=route.name,
        utterances=list(route.utterances),
        description=route.description,
        threshold=route.threshold,
        metadata=dict(route.metadata),
        handler_name=getattr(route.handler, "__name__", None),
        handler_path=route.metadata.get("handler_path"),
        embedded=route.is_embedded,
    )


@router.get("", response_model=list[RouteSummaryResponse], summary="List registered routes")
def list_routes(layer: LayerDependency) -> list[RouteSummaryResponse]:
    return [
        RouteSummaryResponse(
            name=route.name,
            utterance_count=len(route.utterances),
            threshold=route.threshold,
            description=route.description,
        )
        for route in layer.routes
    ]


@router.get("/{name}", response_model=RouteDetailResponse, summary="Get route detail")
def get_route(name: str, layer: LayerDependency) -> RouteDetailResponse:
    route = layer.get(name)
    if route is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="route not found")
    return _to_detail(route)


@router.post(
    "",
    response_model=RouteDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new route",
)
def create_route(
    payload: RouteCreateRequest,
    layer: LayerDependency,
) -> RouteDetailResponse:
    metadata = dict(payload.metadata)
    if payload.handler_path is not None:
        metadata["handler_path"] = payload.handler_path
    handler = _load_handler(payload.handler_path) if payload.handler_path else None
    route = Route(
        name=payload.name,
        utterances=payload.utterances,
        description=payload.description,
        threshold=payload.threshold,
        metadata=metadata,
        handler=handler,
    )
    layer.add(route)
    return _to_detail(route)


@router.put("/{name}", response_model=RouteDetailResponse, summary="Update a route")
def update_route(
    name: str,
    payload: RouteUpdateRequest,
    layer: LayerDependency,
) -> RouteDetailResponse:
    existing = layer.get(name)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="route not found")
    metadata = dict(existing.metadata if payload.metadata is None else payload.metadata)
    handler_path = (
        payload.handler_path
        if payload.handler_path is not None
        else metadata.get("handler_path")
    )
    if handler_path is not None:
        metadata["handler_path"] = handler_path
    handler = _load_handler(handler_path) if isinstance(handler_path, str) else existing.handler
    utterances = (
        list(existing.utterances)
        if payload.utterances is None
        else payload.utterances
    )
    description = (
        existing.description
        if payload.description is None
        else payload.description
    )
    threshold = (
        existing.threshold if payload.threshold is None else payload.threshold
    )
    updated = Route(
        name=name,
        utterances=utterances,
        description=description,
        threshold=threshold,
        metadata=metadata,
        handler=handler,
    )
    layer.update(updated)
    return _to_detail(updated)


@router.delete("/{name}", response_model=OperationStatusResponse, summary="Delete a route")
def delete_route(name: str, layer: LayerDependency) -> OperationStatusResponse:
    route = layer.get(name)
    if route is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="route not found")
    layer.remove(name)
    return OperationStatusResponse(status="ok", detail=f"deleted route '{name}'")


@router.post(
    "/{name}/test",
    response_model=RouteTestResponse,
    summary="Test one route against a query",
)
def test_route(
    name: str,
    payload: RouteTestRequest,
    layer: LayerDependency,
) -> RouteTestResponse:
    route = layer.get(name)
    if route is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="route not found")
    query_vector = layer.encoder.encode_single(payload.query)
    score = route.score(query_vector, strategy=layer.config.routing_strategy)
    threshold = layer.config.resolve_threshold(route.threshold)
    return RouteTestResponse(
        route=name,
        score=score,
        threshold=threshold,
        would_match=score >= threshold,
    )
