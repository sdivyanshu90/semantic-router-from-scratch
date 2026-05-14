"""Unit tests for `semantic_router.index.local`."""

from __future__ import annotations

from semantic_router.index import LocalIndex
from semantic_router.route import Route


def test_local_index_search_returns_highest_scoring_route(mock_encoder) -> None:
    travel = Route(name="travel", utterances=["book a flight"])
    music = Route(name="music", utterances=["play some jazz"])
    travel.embed(mock_encoder)
    music.embed(mock_encoder)
    index = LocalIndex()
    index.build([travel, music])

    results = index.search(mock_encoder.encode_single("book a flight to Paris"), k=1)

    assert results[0][0] == "travel"
