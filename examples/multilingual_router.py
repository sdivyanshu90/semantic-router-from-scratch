"""Cross-lingual semantic routing demo without translation."""

from __future__ import annotations

import hashlib
import os

import numpy as np

from semantic_router import Route, RouteLayer
from semantic_router.encoders import SentenceTransformerEncoder
from semantic_router.encoders.base import BaseEncoder
from semantic_router.utils import normalize


class MultilingualFallbackEncoder(BaseEncoder):
    """Fallback encoder that groups multilingual phrases by intent family."""

    def __init__(self) -> None:
        super().__init__()
        self._anchors = {
            "greeting": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "farewell": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "help_request": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }

    def _category(self, text: str) -> str:
        lowered = text.lower()
        greeting_tokens = {"hello", "hola", "bonjour", "hallo", "namaste", "नमस्ते"}
        if any(token in lowered for token in greeting_tokens):
            return "greeting"
        farewell_tokens = {"bye", "adiós", "au revoir", "tschüss", "अलविदा"}
        if any(token in lowered for token in farewell_tokens):
            return "farewell"
        return "help_request"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, 0.02, size=3).astype(np.float32)
            vectors.append(normalize(self._anchors[self._category(text)] + noise))
        return np.stack(vectors).astype(np.float32, copy=False)

    async def async_encode(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    @property
    def dimensions(self) -> int:
        return 3

    @property
    def name(self) -> str:
        return "multilingual-fallback"


def build_encoder() -> BaseEncoder:
    if os.getenv("SEMANTIC_ROUTER_DOWNLOAD_MODELS") == "1":
        return SentenceTransformerEncoder(
            model_name="paraphrase-multilingual-mpnet-base-v2",
            show_progress=False,
        )
    return MultilingualFallbackEncoder()


def main() -> None:
    layer = RouteLayer(
        routes=[
            Route(
                name="greeting",
                utterances=["hello", "hola", "bonjour", "hallo", "नमस्ते"],
            ),
            Route(
                name="farewell",
                utterances=["goodbye", "adiós", "au revoir", "tschüss", "अलविदा"],
            ),
            Route(
                name="help_request",
                utterances=[
                    "please help me",
                    "necesito ayuda",
                    "j'ai besoin d'aide",
                    "ich brauche hilfe",
                    "मुझे मदद चाहिए",
                ],
            ),
        ],
        encoder=build_encoder(),
    )

    queries = [
        "hello there",
        "hola, necesito ayuda",
        "bonjour mon ami",
        "tschüss und bis bald",
        "मुझे मदद चाहिए",
    ]

    print(f"Using encoder: {layer.encoder.name}\n")
    for query in queries:
        match = layer.route(query)
        matched_name = None if match is None else match.name
        matched_score = None if match is None else round(match.score, 3)
        print(f"{query:<28} -> {matched_name} ({matched_score})")


if __name__ == "__main__":
    main()
