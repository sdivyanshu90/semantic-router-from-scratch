"""Shared pytest fixtures for semantic router tests."""

from __future__ import annotations

import hashlib
from collections.abc import Generator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from semantic_router.config import RouterConfig
from semantic_router.encoders.base import BaseEncoder
from semantic_router.index import LocalIndex
from semantic_router.layer import RouteLayer
from semantic_router.route import Route
from semantic_router.utils import normalize


class MockEncoder(BaseEncoder):
    """Deterministic keyword-aware encoder used for tests."""

    def __init__(self, dimensions: int = 8) -> None:
        super().__init__()
        self._dimensions = dimensions
        self._anchors = {
            "travel": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "weather": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "music": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "finance": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "unknown": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }

    def _category_for_text(self, text: str) -> str:
        lowered = text.lower()
        if any(
            token in lowered
            for token in {"flight", "hotel", "trip", "travel", "seat", "paris"}
        ):
            return "travel"
        if any(
            token in lowered
            for token in {"weather", "rain", "forecast", "sunny", "temperature"}
        ):
            return "weather"
        if any(
            token in lowered
            for token in {"music", "song", "playlist", "jazz", "album", "play"}
        ):
            return "music"
        if any(
            token in lowered
            for token in {"money", "balance", "bank", "transfer", "finance", "card"}
        ):
            return "finance"
        return "unknown"

    def _vector_for_text(self, text: str) -> np.ndarray:
        category = self._category_for_text(text)
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 0.02, size=self._dimensions).astype(np.float32)
        return normalize(self._anchors[category] + noise).astype(np.float32, copy=False)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector_for_text(text) for text in texts]).astype(
            np.float32,
            copy=False,
        )

    async def async_encode(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def name(self) -> str:
        return "mock-encoder"


@pytest.fixture
def mock_encoder() -> MockEncoder:
    """Return the deterministic mock encoder used throughout the test suite."""

    return MockEncoder()


@pytest.fixture
def sample_routes() -> list[Route]:
    """Return four representative routes spanning distinct intent clusters."""

    return [
        Route(
            name="travel",
            utterances=["book a flight", "reserve a hotel", "plan my trip to Paris"],
            description="Travel booking and planning",
            threshold=0.80,
        ),
        Route(
            name="weather",
            utterances=["what is the weather today", "will it rain tomorrow", "show the forecast"],
            description="Weather queries",
        ),
        Route(
            name="music",
            utterances=["play some jazz", "shuffle my playlist", "start the album"],
            description="Music playback",
        ),
        Route(
            name="finance",
            utterances=["check my balance", "transfer money", "show bank transactions"],
            description="Finance requests",
        ),
    ]


@pytest.fixture
def embedded_layer(mock_encoder: MockEncoder, sample_routes: list[Route]) -> RouteLayer:
    """Return a pre-embedded `RouteLayer` backed by the deterministic mock encoder."""

    return RouteLayer(
        routes=sample_routes,
        encoder=mock_encoder,
        config=RouterConfig(default_threshold=0.78, top_k=4),
        index=LocalIndex(),
    )


@pytest.fixture
def api_client(embedded_layer: RouteLayer) -> Generator[TestClient, None, None]:
    """Return a FastAPI `TestClient` bound to the embedded test route layer."""

    app = create_app(route_layer=embedded_layer, cors_origins=["*"])
    with TestClient(app) as client:
        yield client
