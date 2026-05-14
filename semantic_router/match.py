"""Structured result objects returned by semantic routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RouteMatch:
    """
    Represent the outcome of a successful routing decision.

    Algorithm:
        This object is a lightweight immutable-style carrier for the match name,
        score, threshold, and any optional handler output produced by the router.

    Complexity:
        Construction is O(1); serialization is O(R) where R is the number of
        score entries in `all_scores`.

    Args:
        name: Matched route name.
        score: Similarity score assigned to the match.
        threshold: Effective threshold that the route cleared.
        query: Original input query.
        metadata: Route metadata copied into the result for convenience.
        handler_result: Optional value returned by the route handler.
        all_scores: Optional mapping of route names to their similarity scores.

    Example:
        >>> match = RouteMatch(name="travel", score=0.91, threshold=0.8, query="book me a flight")
        >>> match.to_dict()["matched"]
        'travel'
    """

    name: str
    score: float
    threshold: float
    query: str
    metadata: dict[str, Any] = field(default_factory=dict)
    handler_result: Any = None
    all_scores: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the match into an API-friendly dictionary.

        Algorithm:
            Flatten dataclass fields into a mapping aligned with REST responses.

        Complexity:
            O(R) where R is the size of `all_scores`, if provided.

        Returns:
            Dictionary suitable for JSON serialization.

        Example:
            >>> RouteMatch(name="x", score=1.0, threshold=0.5, query="q").to_dict()["score"]
            1.0
        """

        return {
            "matched": self.name,
            "score": self.score,
            "threshold": self.threshold,
            "query": self.query,
            "metadata": self.metadata,
            "handler_result": self.handler_result,
            "all_scores": self.all_scores,
        }
