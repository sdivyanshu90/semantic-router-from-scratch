"""Additional coverage-oriented tests for `semantic_router.layer`."""

from __future__ import annotations

import json

import pytest

from semantic_router.config import RouterConfig
from semantic_router.exceptions import RouteConfigurationError, RouteNotFoundError
from semantic_router.index import LocalIndex
from semantic_router.layer import RouteLayer
from semantic_router.route import Route


def test_layer_empty_routes_have_empty_scores(mock_encoder) -> None:
    layer = RouteLayer(routes=[], encoder=mock_encoder, index=LocalIndex())

    assert layer.score_all("anything") == {}
    assert layer.route("anything") is None


def test_layer_build_match_handles_empty_scores(embedded_layer: RouteLayer) -> None:
    assert embedded_layer._build_match("query", {}, include_scores=False) is None


def test_layer_build_match_respects_threshold(embedded_layer: RouteLayer) -> None:
    assert (
        embedded_layer._build_match(
            "query",
            {"travel": 0.1},
            include_scores=False,
        )
        is None
    )


def test_layer_score_all_returns_all_routes(embedded_layer: RouteLayer) -> None:
    scores = embedded_layer.score_all("play some jazz tonight")

    assert set(scores) == set(embedded_layer.list_routes())
    assert scores["music"] > scores["travel"]


@pytest.mark.asyncio
async def test_async_route_supports_async_handlers(mock_encoder) -> None:
    calls: list[str] = []

    async def handler(query: str) -> str:
        calls.append(query)
        return "async-handled"

    layer = RouteLayer(
        routes=[Route(name="music", utterances=["play some jazz"], handler=handler)],
        encoder=mock_encoder,
        index=LocalIndex(),
    )

    match = await layer.async_route("play some jazz", include_scores=True)

    assert match is not None
    assert match.handler_result == "async-handled"
    assert match.all_scores == {"music": pytest.approx(match.score)}
    assert calls == ["play some jazz"]


def test_batch_route_empty_input_returns_empty_list(embedded_layer: RouteLayer) -> None:
    assert embedded_layer.batch_route([]) == []


def test_batch_route_can_include_scores(embedded_layer: RouteLayer) -> None:
    matches = embedded_layer.batch_route(["play some jazz"], include_scores=True)

    assert matches[0] is not None
    assert matches[0].all_scores is not None
    assert set(matches[0].all_scores) == set(embedded_layer.list_routes())


def test_add_duplicate_route_raises(mock_encoder) -> None:
    layer = RouteLayer(
        routes=[Route(name="music", utterances=["play some jazz"])],
        encoder=mock_encoder,
        index=LocalIndex(),
    )

    with pytest.raises(RouteConfigurationError):
        layer.add(Route(name="music", utterances=["play the album"]))


def test_remove_missing_route_raises(embedded_layer: RouteLayer) -> None:
    with pytest.raises(RouteNotFoundError):
        embedded_layer.remove("missing")


def test_update_missing_route_raises(embedded_layer: RouteLayer) -> None:
    with pytest.raises(RouteNotFoundError):
        embedded_layer.update(Route(name="missing", utterances=["hello"]))


def test_layer_calibrate_updates_thresholds(embedded_layer: RouteLayer) -> None:
    original_threshold = embedded_layer.config.default_threshold

    thresholds = embedded_layer.calibrate(
        [
            ("book a flight to Paris", "travel"),
            ("what is the weather today", "weather"),
            ("play some jazz tonight", "music"),
            ("check my bank balance", "finance"),
            ("nonsense blorb phrase", None),
        ]
    )

    assert set(thresholds) == set(embedded_layer.list_routes())
    assert embedded_layer.config.default_threshold != original_threshold
    assert embedded_layer.get("travel").threshold is not None


def test_suggest_threshold_for_empty_layer_uses_default(mock_encoder) -> None:
    layer = RouteLayer(
        routes=[],
        encoder=mock_encoder,
        config=RouterConfig(default_threshold=0.66),
        index=LocalIndex(),
    )

    assert layer.suggest_threshold() == 0.66


def test_suggest_threshold_for_populated_layer_is_bounded(embedded_layer: RouteLayer) -> None:
    threshold = embedded_layer.suggest_threshold()

    assert embedded_layer.config.calibration_min_threshold <= threshold <= 1.0


def test_load_reembeds_routes_without_saved_vectors(tmp_path, mock_encoder, sample_routes) -> None:
    layer = RouteLayer(routes=sample_routes, encoder=mock_encoder, index=LocalIndex())
    payload = {
        "config": layer.config.to_dict(),
        "routes": [],
    }
    for route in layer.routes:
        route_data = route.to_dict()
        route_data["_centroid"] = None
        route_data["_utterance_vectors"] = []
        payload["routes"].append(route_data)
    path = tmp_path / "router_state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored = RouteLayer(routes=[], encoder=mock_encoder, index=LocalIndex())
    restored.load(str(path))

    assert restored.get("travel") is not None
    assert restored.get("travel").is_embedded
