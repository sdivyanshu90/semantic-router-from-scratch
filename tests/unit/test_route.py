"""Unit tests for `semantic_router.route`."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_router.exceptions import RouteNotEmbeddedError
from semantic_router.route import Route


def test_route_embeds_with_expected_centroid_shape(mock_encoder) -> None:
    route = Route(name="travel", utterances=["book a flight", "reserve a hotel"])

    route.embed(mock_encoder)

    assert route._centroid is not None
    assert route._centroid.shape == (mock_encoder.dimensions,)


def test_route_centroid_is_l2_normalized(mock_encoder) -> None:
    route = Route(name="travel", utterances=["book a flight", "reserve a hotel"])
    route.embed(mock_encoder)

    assert route._centroid is not None
    assert np.isclose(np.linalg.norm(route._centroid), 1.0, atol=1e-6)


def test_route_score_is_bounded(mock_encoder) -> None:
    route = Route(name="travel", utterances=["book a flight", "reserve a hotel"])
    route.embed(mock_encoder)
    query_vector = mock_encoder.encode_single("plan my travel itinerary")

    score = route.score(query_vector)

    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_route_max_strategy_beats_centroid_strategy(mock_encoder) -> None:
    route = Route(name="travel", utterances=["book a flight", "reserve a hotel"])
    route.embed(mock_encoder)
    query_vector = mock_encoder.encode_single("book a flight")

    centroid_score = route.score(query_vector, strategy="centroid")
    max_score = route.score(query_vector, strategy="max")

    assert max_score >= centroid_score


def test_route_serialization_round_trip_preserves_fields(mock_encoder) -> None:
    route = Route(
        name="travel",
        utterances=["book a flight", "reserve a hotel"],
        description="Travel tasks",
        threshold=0.82,
        metadata={"tier": "premium", "handler_path": None},
    )
    route.embed(mock_encoder)

    restored = Route.from_dict(route.to_dict())

    assert restored.name == route.name
    assert restored.utterances == route.utterances
    assert restored.description == route.description
    assert restored.threshold == route.threshold
    assert restored.metadata["tier"] == "premium"
    assert restored.is_embedded


def test_embed_without_utterances_raises(mock_encoder) -> None:
    route = Route(name="empty", utterances=[])

    with pytest.raises(RouteNotEmbeddedError):
        route.embed(mock_encoder)
