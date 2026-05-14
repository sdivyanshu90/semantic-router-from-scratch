"""
Abstract encoder interfaces used by the semantic router.

All encoder implementations must return L2-normalized embeddings. Normalization
matters because cosine similarity between normalized vectors reduces to a simple
dot product. That makes scoring numerically stable, backend-independent, and
cheap enough to use throughout the routing pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from semantic_router.cache import EmbeddingCache


class BaseEncoder(ABC):
    """
    Define the contract shared by all embedding backends.

    Algorithm:
        Concrete subclasses implement batch encoding and async batch encoding.
        This base class provides a convenience single-text wrapper that delegates
        to the batch implementation.

    Complexity:
        `encode_single` is O(D) plus the underlying encoder cost.

    Example:
        >>> class EchoEncoder(BaseEncoder):
        ...     def encode(self, texts: list[str]) -> np.ndarray:
        ...         return np.full((len(texts), 2), 1 / np.sqrt(2), dtype=np.float32)
        ...     async def async_encode(self, texts: list[str]) -> np.ndarray:
        ...         return self.encode(texts)
        ...     @property
        ...     def dimensions(self) -> int:
        ...         return 2
        ...     @property
        ...     def name(self) -> str:
        ...         return "echo"
        >>> EchoEncoder().encode_single("hello").shape
        (2,)
    """

    def __init__(self, cache: EmbeddingCache | None = None) -> None:
        """
        Attach optional cache state to an encoder instance.

        Algorithm:
            Store a reference to the cache so subclasses can opt into cache-aware
            encode paths without changing the interface.

        Complexity:
            O(1).

        Args:
            cache: Optional embedding cache shared across encoder calls.

        Example:
            >>> BaseEncoder.__mro__[0].__name__
            'BaseEncoder'
        """

        self.cache = cache

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an `(N, D)` float32 L2-normalized embedding matrix."""

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode one text and return a one-dimensional normalized vector.

        Algorithm:
            Delegate to the batch `encode` implementation with a singleton list
            and return the first row of the result.

        Complexity:
            O(D) plus the underlying encoder cost.

        Args:
            text: Input text to encode.

        Returns:
            Normalized embedding vector with shape `(D,)`.

        Example:
            >>> class UnitEncoder(BaseEncoder):
            ...     def encode(self, texts: list[str]) -> np.ndarray:
            ...         return np.full(
            ...             (len(texts), 3),
            ...             1 / np.sqrt(3),
            ...             dtype=np.float32,
            ...         )
            ...     async def async_encode(self, texts: list[str]) -> np.ndarray:
            ...         return self.encode(texts)
            ...     @property
            ...     def dimensions(self) -> int:
            ...         return 3
            ...     @property
            ...     def name(self) -> str:
            ...         return "unit"
            >>> UnitEncoder().encode_single("x").shape
            (3,)
        """

        return cast(np.ndarray, self.encode([text])[0])

    @abstractmethod
    async def async_encode(self, texts: list[str]) -> np.ndarray:
        """Asynchronously return an `(N, D)` float32 normalized embedding matrix."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensionality produced by this backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable encoder name for logging and cache keys."""
