"""Minimal semantic router quickstart example."""

from __future__ import annotations

import hashlib
import os

import numpy as np

from semantic_router import Route, RouteLayer
from semantic_router.encoders import SentenceTransformerEncoder
from semantic_router.encoders.base import BaseEncoder
from semantic_router.utils import normalize


class QuickstartFallbackEncoder(BaseEncoder):
    """Small deterministic fallback used when a transformer model is unavailable."""

    def __init__(self) -> None:
        super().__init__()
        self._anchors = {
            "travel": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "weather": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "music": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "other": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }

    def _category(self, text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in {"flight", "hotel", "travel", "trip"}):
            return "travel"
        if any(token in lowered for token in {"weather", "rain", "forecast", "temperature"}):
            return "weather"
        if any(token in lowered for token in {"music", "song", "playlist", "jazz", "play"}):
            return "music"
        return "other"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, 0.02, size=4).astype(np.float32)
            vectors.append(normalize(self._anchors[self._category(text)] + noise))
        return np.stack(vectors).astype(np.float32, copy=False)

    async def async_encode(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    @property
    def dimensions(self) -> int:
        return 4

    @property
    def name(self) -> str:
        return "quickstart-fallback"


def build_encoder() -> BaseEncoder:
    """Prefer the local transformer model but keep the example runnable offline."""

    if os.getenv("SEMANTIC_ROUTER_DOWNLOAD_MODELS") == "1":
        return SentenceTransformerEncoder(
            model_name="all-MiniLM-L6-v2",
            show_progress=False,
        )
    return QuickstartFallbackEncoder()


def main() -> None:
    layer = RouteLayer(
        routes=[
            Route(name="travel", utterances=["book me a flight", "reserve a hotel room"]),
            Route(name="weather", utterances=["will it rain tomorrow", "show me the forecast"]),
            Route(name="music", utterances=["play some jazz", "shuffle my playlist"]),
        ],
        encoder=build_encoder(),
    )

    for query in ["book a flight to Tokyo", "will it rain this weekend", "play upbeat music"]:
        match = layer.route(query)
        matched_name = None if match is None else match.name
        matched_score = None if match is None else round(match.score, 3)
        print(f"{query:<28} -> {matched_name} ({matched_score})")


if __name__ == "__main__":
    main()
