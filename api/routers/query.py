"""Query-time routing endpoints for semantic router inference."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from api.dependencies import get_route_layer
from api.schemas import (
    BatchQueryRequest,
    CalibrationSuggestResponse,
    EmbedRequest,
    EmbedResponse,
    OperationStatusResponse,
    QueryRequest,
    QueryResponse,
    SaveLoadRequest,
)
from semantic_router.layer import RouteLayer

router = APIRouter(tags=["query"])
LayerDependency = Annotated[RouteLayer, Depends(get_route_layer)]


@router.post("/route", response_model=QueryResponse, summary="Route one query")
def route_query(
    payload: QueryRequest,
    request: Request,
    layer: LayerDependency,
) -> QueryResponse:
    match = layer.route(payload.query, include_scores=payload.include_scores)
    request.state.routing_events = [
        {
            "query_length": len(payload.query),
            "matched_route": None if match is None else match.name,
            "score": None if match is None else match.score,
        }
    ]
    if match is None:
        scores = layer.score_all(payload.query) if payload.include_scores else None
        return QueryResponse(
            matched=None,
            score=None,
            handler_result=None,
            all_scores=scores,
        )
    return QueryResponse(
        matched=match.name,
        score=match.score,
        handler_result=match.handler_result,
        all_scores=match.all_scores,
    )


@router.post(
    "/batch-route",
    response_model=list[QueryResponse],
    summary="Route a batch of queries",
)
def batch_route_query(
    payload: BatchQueryRequest,
    request: Request,
    layer: LayerDependency,
) -> list[QueryResponse]:
    matches = layer.batch_route(payload.queries, include_scores=payload.include_scores)
    request.state.routing_events = [
        {
            "query_length": len(query),
            "matched_route": None if match is None else match.name,
            "score": None if match is None else match.score,
        }
        for query, match in zip(payload.queries, matches, strict=True)
    ]
    responses: list[QueryResponse] = []
    for query, match in zip(payload.queries, matches, strict=True):
        if match is None:
            scores = layer.score_all(query) if payload.include_scores else None
            responses.append(
                QueryResponse(
                    matched=None,
                    score=None,
                    handler_result=None,
                    all_scores=scores,
                )
            )
        else:
            responses.append(
                QueryResponse(
                    matched=match.name,
                    score=match.score,
                    handler_result=match.handler_result,
                    all_scores=match.all_scores,
                )
            )
    return responses


@router.post(
    "/embed",
    response_model=EmbedResponse,
    summary="Embed texts with the active encoder",
)
def embed_texts(
    payload: EmbedRequest,
    layer: LayerDependency,
) -> EmbedResponse:
    matrix = layer.encoder.encode(payload.texts)
    return EmbedResponse(vectors=matrix.tolist(), model=layer.encoder.name)


@router.get(
    "/calibrate/suggest",
    response_model=CalibrationSuggestResponse,
    summary="Suggest a threshold for the loaded routes",
)
def suggest_threshold(layer: LayerDependency) -> CalibrationSuggestResponse:
    return CalibrationSuggestResponse(
        suggested_threshold=layer.suggest_threshold(),
        route_count=len(layer.list_routes()),
    )


@router.post("/save", response_model=OperationStatusResponse, summary="Save router state to disk")
def save_router_state(
    payload: SaveLoadRequest,
    layer: LayerDependency,
) -> OperationStatusResponse:
    layer.save(payload.path)
    return OperationStatusResponse(status="ok", detail=f"saved router to {payload.path}")


@router.post("/load", response_model=OperationStatusResponse, summary="Load router state from disk")
def load_router_state(
    payload: SaveLoadRequest,
    layer: LayerDependency,
) -> OperationStatusResponse:
    try:
        layer.load(payload.path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="save file not found",
        ) from exc
    return OperationStatusResponse(status="ok", detail=f"loaded router from {payload.path}")
