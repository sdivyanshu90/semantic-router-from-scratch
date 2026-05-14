"""OpenAI embedding backend for managed vector generation."""

from __future__ import annotations

import os

import httpx
import numpy as np

from semantic_router.encoders.base import BaseEncoder
from semantic_router.exceptions import EncoderError
from semantic_router.utils import normalize


class OpenAIEncoder(BaseEncoder):
    """
    Encode text with the OpenAI embeddings REST API.

    Algorithm:
        1. Check the cache for existing vectors.
        2. Submit only cache misses to `/v1/embeddings`.
        3. Normalize response vectors and merge them into input order.

    Complexity:
        O(N × D) plus remote API latency.

    Args:
        model: OpenAI embedding model name.
        api_key: Optional API key. Falls back to `OPENAI_API_KEY`.
        base_url: Optional API base URL.
        timeout: Request timeout in seconds.
        cache: Optional embedding cache.

    Example:
        >>> OpenAIEncoder().name
        'openai:text-embedding-3-small'
    """

    _MODEL_DIMENSIONS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com",
        timeout: float = 30.0,
        cache: object | None = None,
    ) -> None:
        super().__init__(cache=cache)
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise EncoderError("OPENAI_API_KEY is required for OpenAIEncoder")
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _request_embeddings(self, texts: list[str]) -> np.ndarray:
        response = httpx.post(
            f"{self.base_url}/v1/embeddings",
            headers=self._headers(),
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        ordered = sorted(payload["data"], key=lambda item: item["index"])
        return normalize(np.asarray([item["embedding"] for item in ordered], dtype=np.float32))

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a batch of texts with OpenAI embeddings.

        Algorithm:
            Preserve input ordering by resolving cache hits first and filling only
            the missing positions with vectors returned by the remote API.

        Complexity:
            O(N × D) plus network latency.

        Args:
            texts: Input texts to encode.

        Returns:
            Float32 normalized embedding matrix.

        Example:
            >>> OpenAIEncoder().dimensions in {1536, 3072}
            True
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
        Encode texts asynchronously with the OpenAI embeddings API.

        Algorithm:
            Issue an async HTTP request to the embeddings endpoint and normalize
            the returned vectors.

        Complexity:
            O(N × D) plus remote API latency.

        Args:
            texts: Input texts to encode.

        Returns:
            Float32 normalized embedding matrix.

        Example:
            >>> hasattr(OpenAIEncoder(), "async_encode")
            True
        """

        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/embeddings",
                headers=self._headers(),
                json={"model": self.model, "input": texts},
            )
        response.raise_for_status()
        payload = response.json()
        ordered = sorted(payload["data"], key=lambda item: item["index"])
        return normalize(np.asarray([item["embedding"] for item in ordered], dtype=np.float32))

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
            >>> OpenAIEncoder(model="text-embedding-3-large").dimensions
            3072
        """

        try:
            return self._MODEL_DIMENSIONS[self.model]
        except KeyError as exc:  # pragma: no cover - configuration guard
            raise EncoderError(f"unknown OpenAI embedding model: {self.model}") from exc

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
            >>> OpenAIEncoder().name.startswith("openai:")
            True
        """

        return f"openai:{self.model}"
