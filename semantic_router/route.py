"""
Route dataclass for semantic routing.

Utterance vectors collapse into a centroid like this:

utt_1 ──┐
utt_2 ──┼──► [average] ──► centroid ──► stored as _centroid
utt_3 ──┘
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import structlog

from semantic_router.encoders.base import BaseEncoder
from semantic_router.exceptions import RouteConfigurationError, RouteNotEmbeddedError
from semantic_router.utils import cosine_similarity, normalize

logger = structlog.get_logger(__name__)

RouteHandler = Callable[[str], Any]


def _load_handler(handler_path: str | None) -> RouteHandler | None:
    """
    Resolve a dotted handler path into a Python callable.

    Algorithm:
        Split the `module:function` string, import the module, and return the
        named attribute when it is callable.

    Complexity:
        O(1) ignoring import system cost.

    Args:
        handler_path: Dotted handler reference such as `pkg.module:func`.

    Returns:
        Resolved callable or `None` when no path is supplied.

    Raises:
        RouteConfigurationError: If the path cannot be resolved to a callable.

    Example:
        >>> _load_handler(None) is None
        True
    """

    if handler_path is None:
        return None
    module_name, separator, function_name = handler_path.partition(":")
    if not separator:
        raise RouteConfigurationError(
            "handler_path must use the form 'module:function'"
        )
    module = importlib.import_module(module_name)
    handler = getattr(module, function_name)
    if not callable(handler):
        raise RouteConfigurationError(f"handler '{handler_path}' is not callable")
    return cast(RouteHandler, handler)


@dataclass(slots=True)
class Route:
    """
    Represent a semantic route with example utterances and optional handler.

    Algorithm:
        Each route stores one or more example utterances. The `embed` method turns
        those utterances into normalized vectors and also computes a centroid used
        for fast candidate retrieval. The `score` method compares a query vector to
        the centroid or to the full utterance set depending on the chosen strategy.

    Complexity:
        Embedding is O(U × D) plus encoder cost. Scoring is O(D) for centroid and
        O(U × D) for `max` or `mean`, where U is the utterance count.

    Args:
        name: Unique route identifier.
        utterances: Example phrases that define the route.
        description: Optional human-readable explanation.
        handler: Optional callable invoked when the route matches.
        threshold: Optional route-specific threshold override.
        metadata: Arbitrary JSON-serializable metadata.

    Example:
        >>> route = Route(name="travel", utterances=["book a flight"])
        >>> route.name
        'travel'
    """

    name: str
    utterances: list[str]
    description: str | None = None
    handler: RouteHandler | None = None
    threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _centroid: np.ndarray | None = field(default=None, init=False, repr=False)
    _utterance_vectors: list[np.ndarray] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """
        Validate route fields immediately after construction.

        Algorithm:
            Check name and threshold constraints before the route enters any layer.

        Complexity:
            O(1).

        Raises:
            RouteConfigurationError: If name or threshold is invalid.

        Example:
            >>> Route(name="x", utterances=["hello"]).threshold is None
            True
        """

        if not self.name:
            raise RouteConfigurationError("route name must not be empty")
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise RouteConfigurationError("route threshold must be between 0.0 and 1.0")

    @property
    def is_embedded(self) -> bool:
        """
        Report whether the route has computed embedding state.

        Algorithm:
            A route is considered embedded when both a centroid and utterance
            vectors are available.

        Complexity:
            O(1).

        Returns:
            `True` when embedding state is present; otherwise `False`.

        Example:
            >>> Route(name="x", utterances=["hi"]).is_embedded
            False
        """

        return self._centroid is not None and bool(self._utterance_vectors)

    def embed(self, encoder: BaseEncoder) -> None:
        """
        Embed all route utterances and compute the centroid representation.

        Algorithm:
            1. Encode all utterances with the supplied encoder.
            2. Store every utterance vector for multi-vector scoring.
            3. Average the utterance vectors to form a centroid.
            4. Normalize the centroid for cosine similarity.

        Complexity:
            O(U × D) plus encoder cost.

        Args:
            encoder: Embedding backend used to encode utterances.

        Raises:
            RouteNotEmbeddedError: If no utterances are available to embed.

        Example:
            >>> class ConstantEncoder(BaseEncoder):
            ...     def encode(self, texts: list[str]) -> np.ndarray:
            ...         return normalize(np.ones((len(texts), 2), dtype=np.float32))
            ...     async def async_encode(self, texts: list[str]) -> np.ndarray:
            ...         return self.encode(texts)
            ...     @property
            ...     def dimensions(self) -> int:
            ...         return 2
            ...     @property
            ...     def name(self) -> str:
            ...         return "constant"
            >>> route = Route(name="demo", utterances=["a", "b"])
            >>> route.embed(ConstantEncoder())
            >>> route.is_embedded
            True
        """

        if not self.utterances:
            raise RouteNotEmbeddedError("cannot embed a route without utterances")
        matrix = encoder.encode(self.utterances)
        self._utterance_vectors = [
            vector.astype(np.float32, copy=False) for vector in matrix
        ]
        centroid = np.mean(matrix, axis=0, dtype=np.float32)
        self._centroid = normalize(centroid).astype(np.float32, copy=False)

    def score(self, query_vector: np.ndarray, strategy: str = "centroid") -> float:
        """
        Compute similarity between a query and this route.

        Algorithm:
            centroid  → cosine_similarity(query_vector, self._centroid)
            max       → max(cosine_similarity(query_vector, utt) for utt in utterances)
            mean      → mean(cosine_similarity(query_vector, utt) for utt in utterances)

        Complexity:
            centroid: O(D)        — single dot product
            max/mean: O(U × D)    — U utterances, D dimensions

        Args:
            query_vector: L2-normalized embedding of shape `(D,)`.
            strategy: Aggregation strategy. One of `"centroid"`, `"max"`, `"mean"`.

        Returns:
            Cosine similarity in range `[-1.0, 1.0]`.

        Raises:
            RouteNotEmbeddedError: If the route has not been embedded yet.
            ValueError: If strategy is unsupported.

        Example:
            >>> route = Route(name="travel", utterances=["book a flight"])
            >>> class UnitEncoder(BaseEncoder):
            ...     def encode(self, texts: list[str]) -> np.ndarray:
            ...         return normalize(np.ones((len(texts), 2), dtype=np.float32))
            ...     async def async_encode(self, texts: list[str]) -> np.ndarray:
            ...         return self.encode(texts)
            ...     @property
            ...     def dimensions(self) -> int:
            ...         return 2
            ...     @property
            ...     def name(self) -> str:
            ...         return "unit"
            >>> route.embed(UnitEncoder())
            >>> query_vec = normalize(np.array([1.0, 1.0], dtype=np.float32))
            >>> route.score(query_vec, strategy="centroid")
            1.0
        """

        if not self.is_embedded or self._centroid is None:
            raise RouteNotEmbeddedError(f"route '{self.name}' has not been embedded")
        query = normalize(query_vector).astype(np.float32, copy=False)
        if strategy == "centroid":
            return cosine_similarity(query, self._centroid)
        if strategy == "max":
            return max(
                cosine_similarity(query, vector)
                for vector in self._utterance_vectors
            )
        if strategy == "mean":
            scores = [cosine_similarity(query, vector) for vector in self._utterance_vectors]
            return float(np.mean(scores, dtype=np.float32))
        raise ValueError("strategy must be one of 'centroid', 'max', or 'mean'")

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the route, including cached embeddings, to a dictionary.

        Algorithm:
            Convert public fields to JSON-friendly values and include private
            embeddings as lists so saved router state can reload without recompute.

        Complexity:
            O(U × D) when embeddings are present.

        Returns:
            Serialized route dictionary.

        Example:
            >>> Route(name="x", utterances=["hello"]).to_dict()["name"]
            'x'
        """

        handler_path = self.metadata.get("handler_path")
        return {
            "name": self.name,
            "utterances": list(self.utterances),
            "description": self.description,
            "threshold": self.threshold,
            "metadata": {**self.metadata, "handler_path": handler_path},
            "handler_name": getattr(self.handler, "__name__", None),
            "_centroid": None if self._centroid is None else self._centroid.tolist(),
            "_utterance_vectors": [vector.tolist() for vector in self._utterance_vectors],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Route:
        """
        Reconstruct a route from serialized state.

        Algorithm:
            Rebuild the public dataclass fields, then restore embedding arrays and
            resolve an optional handler path from metadata.

        Complexity:
            O(U × D) when embeddings are present.

        Args:
            data: Serialized route dictionary.

        Returns:
            Reconstructed `Route` instance.

        Example:
            >>> Route.from_dict(
            ...     {"name": "x", "utterances": ["hi"], "metadata": {}}
            ... ).name
            'x'
        """

        metadata = dict(data.get("metadata", {}))
        handler: RouteHandler | None = None
        handler_path = metadata.get("handler_path")
        if isinstance(handler_path, str):
            try:
                handler = _load_handler(handler_path)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning(
                    "route_handler_load_failed",
                    handler_path=handler_path,
                    error=str(exc),
                )
        route = cls(
            name=str(data["name"]),
            utterances=[str(item) for item in data.get("utterances", [])],
            description=data.get("description"),
            handler=handler,
            threshold=data.get("threshold"),
            metadata=metadata,
        )
        centroid = data.get("_centroid")
        if centroid is not None:
            route._centroid = np.asarray(centroid, dtype=np.float32)
        utterance_vectors = data.get("_utterance_vectors", [])
        route._utterance_vectors = [
            np.asarray(vector, dtype=np.float32) for vector in utterance_vectors
        ]
        return route
