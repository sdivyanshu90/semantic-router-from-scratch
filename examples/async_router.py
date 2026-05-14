"""Asynchronous routing throughput comparison demo."""

from __future__ import annotations

import asyncio
import hashlib
import os
import time

import numpy as np

from semantic_router import Route, RouteLayer
from semantic_router.encoders import SentenceTransformerEncoder
from semantic_router.encoders.base import BaseEncoder
from semantic_router.utils import normalize


class AsyncFallbackEncoder(BaseEncoder):
    """Deterministic fallback encoder used for throughput demonstrations."""

    def __init__(self) -> None:
        super().__init__()
        self._anchors = {
            "travel": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "weather": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "music": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }

    def _category(self, text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in {"flight", "trip", "hotel"}):
            return "travel"
        if any(token in lowered for token in {"weather", "rain", "forecast"}):
            return "weather"
        return "music"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, 0.02, size=3).astype(np.float32)
            vectors.append(normalize(self._anchors[self._category(text)] + noise))
        return np.stack(vectors).astype(np.float32, copy=False)

    async def async_encode(self, texts: list[str]) -> np.ndarray:
        await asyncio.sleep(0)
        return self.encode(texts)

    @property
    def dimensions(self) -> int:
        return 3

    @property
    def name(self) -> str:
        return "async-fallback"


def build_encoder() -> BaseEncoder:
    if os.getenv("SEMANTIC_ROUTER_DOWNLOAD_MODELS") == "1":
        return SentenceTransformerEncoder(
            model_name="all-MiniLM-L6-v2",
            show_progress=False,
        )
    return AsyncFallbackEncoder()


async def main() -> None:
    layer = RouteLayer(
        routes=[
            Route(name="travel", utterances=["book a flight", "plan my trip"]),
            Route(name="weather", utterances=["will it rain today", "show me the forecast"]),
            Route(name="music", utterances=["play some jazz", "shuffle my playlist"]),
        ],
        encoder=build_encoder(),
    )
    queries = [
        "book a flight to Berlin",
        "what is the weather tomorrow",
        "play some relaxing music",
        "plan my weekend trip",
        "will it rain this evening",
    ] * 10

    sync_started = time.perf_counter()
    sync_matches = [layer.route(query) for query in queries]
    sync_elapsed = time.perf_counter() - sync_started

    async_started = time.perf_counter()
    async_matches = await asyncio.gather(*(layer.async_route(query) for query in queries))
    async_elapsed = time.perf_counter() - async_started

    print(f"Encoder: {layer.encoder.name}")
    print(f"Sync throughput : {len(sync_matches) / sync_elapsed:.2f} queries/second")
    print(f"Async throughput: {len(async_matches) / async_elapsed:.2f} queries/second")


if __name__ == "__main__":
    asyncio.run(main())
