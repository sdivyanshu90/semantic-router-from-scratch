"""Cohere embedding backend for managed vector generation."""

from __future__ import annotations

import os

import httpx
import numpy as np

from semantic_router.encoders.base import BaseEncoder
from semantic_router.exceptions import EncoderError
from semantic_router.utils import normalize


class CohereEncoder(BaseEncoder):
    """
    Encode text with Cohere's embedding API.

    Algorithm:
        Submit the input batch to Cohere's embedding endpoint, normalize the
        returned vectors, and expose the result through the same interface as
        local encoders.

    Complexity:
        O(N × D) plus remote API latency.

    Args:
        model: Cohere embedding model.
        api_key: Optional API key. Falls back to `COHERE_API_KEY`.
        base_url: Optional API base URL.
        input_type: Cohere input type.
        timeout: Request timeout in seconds.
        cache: Optional embedding cache.

    Example:
        >>> CohereEncoder().name.startswith("cohere:")
        True
    """

    _MODEL_DIMENSIONS: dict[str, int] = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
    }

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: str | None = None,
        base_url: str = "https://api.cohere.com",
        input_type: str = "search_document",
        timeout: float = 30.0,
        cache: object | None = None,
    ) -> None:
        super().__init__(cache=cache)
        self.model = model
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.input_type = input_type
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise EncoderError("COHERE_API_KEY is required for CohereEncoder")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _request_embeddings(self, texts: list[str]) -> np.ndarray:
        response = httpx.post(
            f"{self.base_url}/v2/embed",
            headers=self._headers(),
            json={
                "model": self.model,
                "input_type": self.input_type,
                "texts": texts,
                "embedding_types": ["float"],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        vectors = payload["embeddings"]["float"]
        return normalize(np.asarray(vectors, dtype=np.float32))

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a batch of texts with Cohere embeddings.

        Algorithm:
            Resolve cache hits first, then fill the remaining slots with vectors
            produced by the remote embedding service.

        Complexity:
            O(N × D) plus network latency.

        Args:
            texts: Input texts to encode.

        Returns:
            Float32 normalized embedding matrix.

        Example:
            >>> CohereEncoder().dimensions
            1024
        """

        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        cached_vectors: dict[int, np.ndarray] = {}
        missing_texts: list[str] = []
        missing_indices: list[int] = []
        if self.cache is not None:
            for index, text in enumerate(texts):
                cached = self.cache.get(text, self.name)
                if cached is None:
                    missing_texts.append(text)
                    missing_indices.append(index)
                else:
                    cached_vectors[index] = cached.astype(np.float32, copy=False)
        else:
            missing_texts = list(texts)
            missing_indices = list(range(len(texts)))
        if missing_texts:
            vectors = self._request_embeddings(missing_texts)
            for index, vector in zip(missing_indices, vectors, strict=True):
                cached_vectors[index] = vector
                if self.cache is not None:
                    self.cache.set(texts[index], self.name, vector)
        return np.stack([cached_vectors[index] for index in range(len(texts))]).astype(
            np.float32,
            copy=False,
        )

    async def async_encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts asynchronously with Cohere embeddings.

        Algorithm:
            Use `httpx.AsyncClient` to send a non-blocking request to the remote
            embedding service and normalize the returned vectors.

        Complexity:
            O(N × D) plus network latency.

        Args:
            texts: Input texts to encode.

        Returns:
            Float32 normalized embedding matrix.

        Example:
            >>> hasattr(CohereEncoder(), "async_encode")
            True
        """

        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v2/embed",
                headers=self._headers(),
                json={
                    "model": self.model,
                    "input_type": self.input_type,
                    "texts": texts,
                    "embedding_types": ["float"],
                },
            )
        response.raise_for_status()
        payload = response.json()
        return normalize(np.asarray(payload["embeddings"]["float"], dtype=np.float32))

    @property
    def dimensions(self) -> int:
        """
        Return the expected embedding dimensionality for the configured model.

        Algorithm:
            Resolve the model name from a small static map.

        Complexity:
            O(1).

        Returns:
            Embedding dimensionality.

        Raises:
            EncoderError: If the model is unknown.

        Example:
            >>> CohereEncoder().dimensions
            1024
        """

        try:
            return self._MODEL_DIMENSIONS[self.model]
        except KeyError as exc:  # pragma: no cover - configuration guard
            raise EncoderError(f"unknown Cohere embedding model: {self.model}") from exc

    @property
    def name(self) -> str:
        """
        Return a stable backend identifier.

        Algorithm:
            Prefix the model name with the provider family.

        Complexity:
            O(1).

        Returns:
            Stable backend name.

        Example:
            >>> CohereEncoder().name
            'cohere:embed-english-v3.0'
        """

        return f"cohere:{self.model}"
