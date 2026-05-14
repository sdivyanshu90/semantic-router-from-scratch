"""In-memory NumPy index used by default for route candidate retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from semantic_router.exceptions import RouteNotEmbeddedError
from semantic_router.utils import cosine_similarity_matrix, normalize, top_k_indices

if TYPE_CHECKING:
    from semantic_router.route import Route


class BaseIndex(ABC):
    """Abstract vector index interface consumed by `RouteLayer`."""

    @abstractmethod
    def build(self, routes: Sequence[Route]) -> None:
        """Build the index from a collection of embedded routes."""

    @abstractmethod
    def add(self, route: Route) -> None:
        """Add one embedded route to the index."""

    @abstractmethod
    def remove(self, name: str) -> None:
        """Remove one route from the index."""

    @abstractmethod
    def update(self, route: Route) -> None:
        """Update one route in the index."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return the top-k route names with approximate similarity scores."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all indexed vectors."""


class LocalIndex(BaseIndex):
    """
    Store route centroids in memory and search them with vectorized NumPy.

    Algorithm:
        Maintain a dense centroid matrix aligned with a name list so top-k search
        reduces to one cosine similarity matrix multiplication per query.

    Complexity:
        Search is O(R × D), where R is route count and D is dimensionality.

    Example:
        >>> index = LocalIndex()
        >>> index.search(np.array([1.0], dtype=np.float32), 1)
        []
    """

    def __init__(self) -> None:
        self._names: list[str] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)

    def _rebuild(self, routes: Sequence[Route]) -> None:
        self._names = []
        vectors: list[np.ndarray] = []
        for route in routes:
            if route._centroid is None:
                raise RouteNotEmbeddedError(
                    f"route '{route.name}' must be embedded before indexing"
                )
            self._names.append(route.name)
            vectors.append(route._centroid.astype(np.float32, copy=False))
        if vectors:
            self._matrix = np.vstack(vectors).astype(np.float32, copy=False)
        else:
            self._matrix = np.zeros((0, 0), dtype=np.float32)

    def build(self, routes: Sequence[Route]) -> None:
        self._rebuild(routes)

    def add(self, route: Route) -> None:
        if route._centroid is None:
            raise RouteNotEmbeddedError(
                f"route '{route.name}' must be embedded before indexing"
            )
        if route.name in self._names:
            self.update(route)
            return
        self._names.append(route.name)
        row = route._centroid.astype(np.float32, copy=False)[np.newaxis, :]
        if self._matrix.size == 0:
            self._matrix = row
        else:
            self._matrix = np.vstack([self._matrix, row]).astype(np.float32, copy=False)

    def remove(self, name: str) -> None:
        if name not in self._names:
            return
        index = self._names.index(name)
        self._names.pop(index)
        if self._matrix.shape[0] == 1:
            self._matrix = np.zeros((0, self._matrix.shape[1]), dtype=np.float32)
        else:
            self._matrix = np.delete(self._matrix, index, axis=0)

    def update(self, route: Route) -> None:
        if route._centroid is None:
            raise RouteNotEmbeddedError(
                f"route '{route.name}' must be embedded before indexing"
            )
        if route.name not in self._names:
            self.add(route)
            return
        index = self._names.index(route.name)
        self._matrix[index] = route._centroid.astype(np.float32, copy=False)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self._names or self._matrix.size == 0:
            return []
        normalized_query = normalize(query_vector).astype(np.float32, copy=False)
        scores = cosine_similarity_matrix(normalized_query[np.newaxis, :], self._matrix)[0]
        indices = top_k_indices(scores, k)
        return [(self._names[index], float(scores[index])) for index in indices]

    def clear(self) -> None:
        self._names = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
