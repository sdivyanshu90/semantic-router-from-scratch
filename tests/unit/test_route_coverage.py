"""Additional coverage-oriented tests for `semantic_router.route`."""

from __future__ import annotations

import pytest

from semantic_router.exceptions import RouteConfigurationError
from semantic_router.route import Route, _load_handler


def test_load_handler_rejects_invalid_format() -> None:
    with pytest.raises(RouteConfigurationError):
        _load_handler("invalid_handler_path")


def test_load_handler_rejects_non_callable_target() -> None:
    with pytest.raises(RouteConfigurationError):
        _load_handler("math:pi")


def test_route_rejects_empty_name() -> None:
    with pytest.raises(RouteConfigurationError):
        Route(name="", utterances=["hello"])


def test_route_rejects_invalid_threshold() -> None:
    with pytest.raises(RouteConfigurationError):
        Route(name="demo", utterances=["hello"], threshold=1.5)


def test_route_mean_strategy_and_serialized_vectors(mock_encoder) -> None:
    route = Route(name="travel", utterances=["book a flight", "reserve a hotel"])
    route.embed(mock_encoder)

    score = route.score(mock_encoder.encode_single("book a flight to Paris"), strategy="mean")
    payload = route.to_dict()

    assert -1.0 <= score <= 1.0
    assert payload["_centroid"] is not None
    assert len(payload["_utterance_vectors"]) == 2
