"""Additional coverage-oriented tests for `semantic_router.index.local`."""

from __future__ import annotations

from semantic_router.index import LocalIndex
from semantic_router.route import Route


def test_local_index_add_update_remove_and_clear(mock_encoder) -> None:
    travel = Route(name="travel", utterances=["book a flight"])
    weather = Route(name="weather", utterances=["show the forecast"])
    travel.embed(mock_encoder)
    weather.embed(mock_encoder)
    index = LocalIndex()

    index.add(travel)
    index.add(weather)
    assert index.search(mock_encoder.encode_single("show the forecast"), k=2)

    updated_travel = Route(name="travel", utterances=["plan my trip to Paris"])
    updated_travel.embed(mock_encoder)
    index.update(updated_travel)
    assert (
        index.search(mock_encoder.encode_single("plan my trip"), k=1)[0][0]
        == "travel"
    )

    index.remove("weather")
    assert all(
        name != "weather"
        for name, _score in index.search(
            mock_encoder.encode_single("show the forecast"),
            2,
        )
    )

    index.remove("missing")
    index.clear()
    assert index.search(mock_encoder.encode_single("book a flight"), k=1) == []


def test_local_index_add_existing_route_updates(mock_encoder) -> None:
    route = Route(name="travel", utterances=["book a flight"])
    route.embed(mock_encoder)
    index = LocalIndex()
    index.add(route)

    replacement = Route(name="travel", utterances=["reserve a hotel"])
    replacement.embed(mock_encoder)
    index.add(replacement)

    assert index.search(mock_encoder.encode_single("reserve a hotel"), k=1)[0][0] == "travel"
