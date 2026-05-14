"""Pydantic request and response models for the semantic router API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RouteCreateRequest(BaseModel):
    """Payload used to create a new semantic route through the API."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "travel",
                "utterances": ["book me a flight", "reserve a hotel room"],
                "description": "Travel booking requests",
                "threshold": 0.81,
                "metadata": {"team": "growth"},
                "handler_path": None,
            }
        }
    )

    name: str = Field(description="Unique route identifier.")
    utterances: list[str] = Field(description="Example phrases defining the route.")
    description: str | None = Field(
        default=None,
        description="Human-readable route description.",
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional route-specific threshold override.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary JSON metadata.",
    )
    handler_path: str | None = Field(
        default=None,
        description="Optional handler reference in module:function format.",
    )

    @field_validator("utterances")
    @classmethod
    def validate_utterances(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("utterances must contain at least one example")
        return value


class RouteUpdateRequest(BaseModel):
    """Payload used to update an existing semantic route."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "utterances": ["can you refund my order", "start a return"],
                "threshold": 0.84,
                "description": "Returns and refund requests",
                "metadata": {"priority": "high"},
                "handler_path": None,
            }
        }
    )

    utterances: list[str] | None = Field(
        default=None,
        description="Replacement utterance set.",
    )
    description: str | None = Field(
        default=None,
        description="Updated human-readable description.",
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Updated route-specific threshold override.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Replacement metadata dictionary.",
    )
    handler_path: str | None = Field(
        default=None,
        description="Optional handler reference in module:function format.",
    )

    @field_validator("utterances")
    @classmethod
    def validate_optional_utterances(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and not value:
            raise ValueError("utterances must contain at least one example when provided")
        return value


class RouteSummaryResponse(BaseModel):
    """Compact route summary returned by list endpoints."""

    name: str = Field(description="Route name.")
    utterance_count: int = Field(description="Number of sample utterances in the route.")
    threshold: float | None = Field(
        description="Route-specific threshold override when configured.",
    )
    description: str | None = Field(description="Human-readable route description.")


class RouteDetailResponse(BaseModel):
    """Full route detail returned by CRUD endpoints."""

    name: str = Field(description="Route name.")
    utterances: list[str] = Field(description="Sample utterances for the route.")
    description: str | None = Field(description="Human-readable route description.")
    threshold: float | None = Field(description="Route-specific threshold override.")
    metadata: dict[str, Any] = Field(description="Arbitrary route metadata.")
    handler_name: str | None = Field(
        description="Resolved Python handler name when configured.",
    )
    handler_path: str | None = Field(
        description="Configured import path for the route handler.",
    )
    embedded: bool = Field(description="Whether the route has computed embeddings.")


class RouteTestRequest(BaseModel):
    """Payload used to test a single route against one query."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"query": "book me a flight to NYC"}}
    )

    query: str = Field(description="Query text to score against a specific route.")


class RouteTestResponse(BaseModel):
    """Detailed scoring response for one route and one query."""

    route: str = Field(description="Route that was evaluated.")
    score: float = Field(description="Cosine similarity score.")
    threshold: float = Field(description="Effective threshold for the route.")
    would_match: bool = Field(description="Whether the score clears the threshold.")


class QueryRequest(BaseModel):
    """Single-query routing request."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"query": "book me a flight", "include_scores": True}
        }
    )

    query: str = Field(description="Natural-language query to route.")
    include_scores: bool = Field(
        default=False,
        description="Whether to include per-route similarity scores in the response.",
    )


class QueryResponse(BaseModel):
    """Single-query routing response."""

    matched: str | None = Field(
        description="Matched route name, or null when no route clears threshold.",
    )
    score: float | None = Field(
        description="Best similarity score, or null when unmatched.",
    )
    handler_result: Any = Field(
        description="Optional handler return value when a route handler ran.",
    )
    all_scores: dict[str, float] | None = Field(
        description="Optional per-route score mapping when requested.",
        default=None,
    )


class BatchQueryRequest(BaseModel):
    """Batch routing request."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "queries": ["book me a flight", "how do I reset my password"],
                "include_scores": False,
            }
        }
    )

    queries: list[str] = Field(description="Queries to route in one batch request.")
    include_scores: bool = Field(
        default=False,
        description="Whether to include per-route similarity scores in each response.",
    )

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("queries must contain at least one entry")
        return value


class EmbedRequest(BaseModel):
    """Embedding request payload."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"texts": ["hello", "goodbye"]}}
    )

    texts: list[str] = Field(description="Texts to encode into embedding vectors.")


class EmbedResponse(BaseModel):
    """Embedding response payload."""

    vectors: list[list[float]] = Field(description="L2-normalized embedding vectors.")
    model: str = Field(description="Encoder backend name that produced the vectors.")


class SaveLoadRequest(BaseModel):
    """Payload used to save or load router state from disk."""

    model_config = ConfigDict(
        json_schema_extra={"example": {"path": "/tmp/router_state.json"}}
    )

    path: str = Field(description="Filesystem path for saved router state.")


class OperationStatusResponse(BaseModel):
    """Generic success response for non-query operations."""

    status: str = Field(description="Operation result status string.")
    detail: str = Field(description="Human-readable status detail.")


class CalibrationSuggestResponse(BaseModel):
    """Suggested threshold response."""

    suggested_threshold: float = Field(
        description="Suggested router threshold based on loaded routes.",
    )
    route_count: int = Field(
        description="Number of routes considered when generating the suggestion.",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status indicator.")
    routes_loaded: int = Field(description="Number of routes currently loaded in memory.")
