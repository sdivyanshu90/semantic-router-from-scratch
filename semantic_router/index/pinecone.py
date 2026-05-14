"""Optional Pinecone-backed index for large-scale route retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from semantic_router.exceptions import EncoderError, RouteNotEmbeddedError
from semantic_router.index.local import BaseIndex

if TYPE_CHECKING:
    from semantic_router.route import Route


class PineconeIndex(BaseIndex):
    """
    Store route centroids in a managed Pinecone vector index.

    Algorithm:
        Upsert route centroids into Pinecone and delegate top-k similarity search
        to the remote vector database.

    Complexity:
        Dominated by remote network and service latency.

    Example:
        >>> PineconeIndex.__name__
        'PineconeIndex'
    """

    def __init__(self, index_name: str, namespace: str = "semantic-router") -> None:
        try:
            from pinecone import Pinecone
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise EncoderError("pinecone-client is required for PineconeIndex") from exc
        self._client: Any = Pinecone()
        self._index = self._client.Index(index_name)
        self.namespace = namespace

    def build(self, routes: Sequence[Route]) -> None:
        for route in routes:
            self.add(route)

    def add(self, route: Route) -> None:
        if route._centroid is None:
            raise RouteNotEmbeddedError(
                f"route '{route.name}' must be embedded before indexing"
            )
        self._index.upsert(
            vectors=[
                {
                    "id": route.name,
                    "values": route._centroid.tolist(),
                    "metadata": {"route_name": route.name},
                }
            ],
            namespace=self.namespace,
        )

    def remove(self, name: str) -> None:
        self._index.delete(ids=[name], namespace=self.namespace)

    def update(self, route: Route) -> None:
        self.add(route)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        result = self._index.query(
            vector=query_vector.tolist(),
            top_k=k,
            namespace=self.namespace,
            include_metadata=True,
        )
        matches = getattr(result, "matches", [])
        return [(match.id, float(match.score)) for match in matches]

    def clear(self) -> None:
        self._index.delete(delete_all=True, namespace=self.namespace)
