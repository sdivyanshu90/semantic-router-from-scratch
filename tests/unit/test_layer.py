"""Unit tests for `semantic_router.layer`."""

from __future__ import annotations

from pathlib import Path

from semantic_router.config import RouterConfig
from semantic_router.index import LocalIndex
from semantic_router.layer import RouteLayer
from semantic_router.route import Route


def test_route_returns_correct_match_for_unambiguous_query(embedded_layer: RouteLayer) -> None:
    match = embedded_layer.route("book me a flight to Paris")

    assert match is not None
    assert match.name == "travel"


def test_route_returns_none_when_scores_below_threshold(embedded_layer: RouteLayer) -> None:
    match = embedded_layer.route("zqxj blorb snazzle frobnicate")

    assert match is None


def test_batch_route_preserves_input_length(embedded_layer: RouteLayer) -> None:
    matches = embedded_layer.batch_route(["book a flight", "play some jazz", "unknown blorb"])

    assert len(matches) == 3


def test_add_increases_route_count(mock_encoder, sample_routes) -> None:
    layer = RouteLayer(routes=sample_routes[:1], encoder=mock_encoder, index=LocalIndex())
    before = len(layer.list_routes())

    layer.add(Route(name="weather", utterances=["show the weather forecast"]))

    assert len(layer.list_routes()) == before + 1


def test_remove_decreases_route_count(embedded_layer: RouteLayer) -> None:
    before = len(embedded_layer.list_routes())

    embedded_layer.remove("finance")

    assert len(embedded_layer.list_routes()) == before - 1


def test_save_load_round_trip_preserves_routes_and_thresholds(
    tmp_path: Path,
    mock_encoder,
    sample_routes,
) -> None:
    layer = RouteLayer(routes=sample_routes, encoder=mock_encoder, index=LocalIndex())
    path = tmp_path / "router.json"

    layer.save(str(path))

    restored = RouteLayer(
        routes=[],
        encoder=mock_encoder,
        config=RouterConfig(default_threshold=0.1),
    )
    restored.load(str(path))

    assert restored.list_routes() == layer.list_routes()
    assert restored.get("travel") is not None
    assert restored.get("travel").threshold == layer.get("travel").threshold


def test_handler_is_called_on_match(mock_encoder) -> None:
    calls: list[str] = []

    def handler(query: str) -> str:
        calls.append(query)
        return "handled"

    layer = RouteLayer(
        routes=[Route(name="music", utterances=["play some jazz"], handler=handler)],
        encoder=mock_encoder,
        index=LocalIndex(),
    )

    match = layer.route("play some jazz")

    assert match is not None
    assert match.handler_result == "handled"
    assert calls == ["play some jazz"]
