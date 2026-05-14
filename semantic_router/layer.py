"""
Core semantic routing engine.

```mermaid
flowchart TD
  A[User Query String] --> B[Encoder.encode]
  B --> C[Query Vector 768-dim]
  C --> D[Index.top_k_search]
  D --> E{Candidates Found?}
  E -- No --> F[Return None]
  E -- Yes --> G[Score Each Route]
  G --> H{Max Score > Threshold?}
  H -- No --> F
  H -- Yes --> I[RouteMatch]
  I --> J{Handler defined?}
  J -- Yes --> K[Execute handler]
  J -- No --> L[Return match only]
  K --> L
```
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from semantic_router.config import RouterConfig
from semantic_router.encoders import BaseEncoder, SentenceTransformerEncoder
from semantic_router.exceptions import RouteConfigurationError, RouteNotFoundError
from semantic_router.index import BaseIndex, LocalIndex
from semantic_router.match import RouteMatch
from semantic_router.route import Route

logger = structlog.get_logger(__name__)


class RouteLayer:
    """
    Route natural-language queries to the most semantically similar route.

    Algorithm:
        Build embeddings for every route, keep a vector index over route centroids,
        retrieve top-k candidates for each query, then compute exact route scores
        with the configured strategy before applying thresholds.

    Complexity:
        Query routing is O(E + K × U × D), where E is query encoding cost, K is
        candidate count, U is utterances per candidate route, and D is dimensions.

    Example:
        >>> layer = RouteLayer(
        ...     routes=[],
        ...     encoder=SentenceTransformerEncoder(show_progress=False),
        ... )
        >>> layer.list_routes()
        []
    """

    def __init__(
        self,
        routes: Sequence[Route] | None = None,
        encoder: BaseEncoder | None = None,
        config: RouterConfig | None = None,
        index: BaseIndex | None = None,
    ) -> None:
        self.encoder = encoder or SentenceTransformerEncoder(show_progress=False)
        self.config = config or RouterConfig()
        self.index = index or LocalIndex()
        self._routes: dict[str, Route] = {}
        for route in routes or []:
            self.add(route)
        self._rebuild_index()

    @property
    def routes(self) -> list[Route]:
        """
        Return routes in stable insertion order.

        Algorithm:
            Materialize the internal route mapping values as a list.

        Complexity:
            O(R).

        Returns:
            List of registered routes.

        Example:
            >>> RouteLayer(
            ...     routes=[],
            ...     encoder=SentenceTransformerEncoder(show_progress=False),
            ... ).routes
            []
        """

        return list(self._routes.values())

    def _ensure_embedded(self, route: Route) -> None:
        if not route.is_embedded:
            route.embed(self.encoder)

    def _rebuild_index(self) -> None:
        self.index.clear()
        self.index.build(self.routes)

    def _candidate_names(
        self,
        query_vector: np.ndarray,
        k: int | None = None,
    ) -> list[str]:
        search_results = self.index.search(query_vector, k or self.config.top_k)
        return [name for name, _score in search_results]

    def _score_routes_for_vector(
        self,
        query_vector: np.ndarray,
        candidate_names: Sequence[str] | None = None,
    ) -> dict[str, float]:
        names = list(candidate_names) if candidate_names is not None else list(self._routes)
        return {
            name: self._routes[name].score(query_vector, strategy=self.config.routing_strategy)
            for name in names
        }

    def score_all(self, query: str) -> dict[str, float]:
        """
        Score a query against every route without applying thresholds.

        Algorithm:
            Encode the query once and evaluate it against the full route set.

        Complexity:
            O(E + R × U × D).

        Args:
            query: Input query string.

        Returns:
            Mapping of route names to similarity scores.

        Example:
            >>> layer = RouteLayer(
            ...     routes=[],
            ...     encoder=SentenceTransformerEncoder(show_progress=False),
            ... )
            >>> layer.score_all("hello")
            {}
        """

        if not self._routes:
            return {}
        query_vector = self.encoder.encode_single(query)
        return self._score_routes_for_vector(query_vector)

    def _build_match(
        self,
        query: str,
        scores: dict[str, float],
        include_scores: bool = False,
        execute_handler: bool = True,
    ) -> RouteMatch | None:
        if not scores:
            return None
        best_name, best_score = max(scores.items(), key=lambda item: item[1])
        route = self._routes[best_name]
        threshold = self.config.resolve_threshold(route.threshold)
        if best_score < threshold:
            return None
        handler_result: Any = None
        handler = route.handler
        if (
            handler is not None
            and execute_handler
            and not inspect.iscoroutinefunction(handler)
        ):
            handler_result = handler(query)
        return RouteMatch(
            name=best_name,
            score=float(best_score),
            threshold=threshold,
            query=query,
            metadata=dict(route.metadata),
            handler_result=handler_result,
            all_scores=scores if include_scores else None,
        )

    def route(self, query: str, include_scores: bool = False) -> RouteMatch | None:
        """
        Route a single query to the best matching semantic route.

        Algorithm:
            1. Encode the query string into one normalized vector.
            2. Retrieve the top-k centroid candidates from the index.
            3. Score each candidate route with the configured strategy.
            4. Compare the best score with the effective threshold.
            5. Return a `RouteMatch` or `None`.

        Complexity:
            O(E + K × U × D).

        Args:
            query: User query string.
            include_scores: Whether to include per-route scores in the result.

        Returns:
            Best `RouteMatch` when a route clears threshold, otherwise `None`.

        Example:
            >>> layer = RouteLayer(
            ...     routes=[],
            ...     encoder=SentenceTransformerEncoder(show_progress=False),
            ... )
            >>> layer.route("hello") is None
            True
        """

        started_at = time.perf_counter()

        # Step 1: encode the query string once into the router's embedding space.
        query_vector = self.encoder.encode_single(query)

        # Step 2: narrow the search space with the centroid index before exact scoring.
        candidate_names = self._candidate_names(query_vector)
        if not candidate_names:
            logger.debug("route_completed", query=query, matched_route=None, latency_ms=0.0)
            return None

        # Step 3: compute exact route scores only for the retrieved candidate set.
        candidate_scores = self._score_routes_for_vector(query_vector, candidate_names)
        final_scores = (
            self._score_routes_for_vector(query_vector)
            if include_scores
            else candidate_scores
        )

        # Step 4: pick the best route and apply its effective threshold.
        match = self._build_match(
            query,
            final_scores,
            include_scores=include_scores,
            execute_handler=True,
        )

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.debug(
            "route_completed",
            query=query,
            matched_route=None if match is None else match.name,
            score=None if match is None else match.score,
            latency_ms=round(elapsed_ms, 3),
        )
        return match

    async def async_route(
        self,
        query: str,
        include_scores: bool = False,
    ) -> RouteMatch | None:
        """
        Route one query asynchronously.

        Algorithm:
            Use the encoder's async interface, then reuse the same candidate search
            and scoring logic as the synchronous path.

        Complexity:
            O(E + K × U × D) plus async scheduling overhead.

        Args:
            query: User query string.
            include_scores: Whether to include per-route scores in the result.

        Returns:
            Best `RouteMatch` or `None`.

        Example:
            >>> hasattr(RouteLayer, "async_route")
            True
        """

        started_at = time.perf_counter()
        matrix = await self.encoder.async_encode([query])
        query_vector = matrix[0]
        candidate_names = self._candidate_names(query_vector)
        if not candidate_names:
            return None
        candidate_scores = self._score_routes_for_vector(query_vector, candidate_names)
        final_scores = (
            self._score_routes_for_vector(query_vector)
            if include_scores
            else candidate_scores
        )
        match = self._build_match(
            query,
            final_scores,
            include_scores=include_scores,
            execute_handler=False,
        )
        if match is not None:
            route = self._routes[match.name]
            handler = route.handler
            if handler is not None and inspect.iscoroutinefunction(handler):
                match.handler_result = await handler(query)
            elif handler is not None:
                match.handler_result = handler(query)
        logger.debug(
            "async_route_completed",
            query=query,
            matched_route=None if match is None else match.name,
            score=None if match is None else match.score,
            latency_ms=round((time.perf_counter() - started_at) * 1000.0, 3),
        )
        return match

    def batch_route(
        self,
        queries: list[str],
        include_scores: bool = False,
    ) -> list[RouteMatch | None]:
        """
        Route many queries in one encoder call.

        Algorithm:
            Encode the entire query batch at once, then evaluate each query vector
            independently against the candidate index.

        Complexity:
            O(Q × (K × U × D)) plus one batch encoding call.

        Args:
            queries: Query strings to route.
            include_scores: Whether to include per-route scores in each match.

        Returns:
            List of matches aligned to the input order.

        Example:
            >>> RouteLayer(
            ...     routes=[],
            ...     encoder=SentenceTransformerEncoder(show_progress=False),
            ... ).batch_route([])
            []
        """

        if not queries:
            return []
        vectors = self.encoder.encode(queries)
        matches: list[RouteMatch | None] = []
        for query, vector in zip(queries, vectors, strict=True):
            candidate_names = self._candidate_names(vector)
            if not candidate_names:
                matches.append(None)
                continue
            candidate_scores = self._score_routes_for_vector(vector, candidate_names)
            final_scores = (
                self._score_routes_for_vector(vector)
                if include_scores
                else candidate_scores
            )
            matches.append(
                self._build_match(
                    query,
                    final_scores,
                    include_scores=include_scores,
                    execute_handler=True,
                )
            )
        return matches

    def add(self, route: Route) -> None:
        """
        Add a route to the layer and embed it if needed.

        Algorithm:
            Validate uniqueness, ensure embeddings exist, then register the route
            in both the route map and candidate index.

        Complexity:
            O(U × D) plus encoder cost when embedding is required.

        Args:
            route: Route to add.

        Raises:
            RouteConfigurationError: If a route with the same name already exists.

        Example:
            >>> layer = RouteLayer(
            ...     routes=[],
            ...     encoder=SentenceTransformerEncoder(show_progress=False),
            ... )
            >>> layer.list_routes()
            []
        """

        if route.name in self._routes:
            raise RouteConfigurationError(f"route '{route.name}' already exists")
        self._ensure_embedded(route)
        self._routes[route.name] = route
        self.index.add(route)

    def remove(self, name: str) -> None:
        """
        Remove a route from the layer.

        Algorithm:
            Delete the route from the internal mapping and remove its centroid from
            the vector index.

        Complexity:
            O(R) for local index maintenance.

        Args:
            name: Route name to delete.

        Raises:
            RouteNotFoundError: If the route does not exist.

        Example:
            >>> hasattr(RouteLayer, "remove")
            True
        """

        if name not in self._routes:
            raise RouteNotFoundError(f"route '{name}' not found")
        del self._routes[name]
        self.index.remove(name)

    def update(self, route: Route) -> None:
        """
        Replace an existing route with new configuration or utterances.

        Algorithm:
            Re-embed the supplied route when needed, overwrite the stored route,
            and refresh its vector index entry.

        Complexity:
            O(U × D) plus encoder cost when embedding is required.

        Args:
            route: Updated route definition.

        Raises:
            RouteNotFoundError: If the route does not already exist.

        Example:
            >>> hasattr(RouteLayer, "update")
            True
        """

        if route.name not in self._routes:
            raise RouteNotFoundError(f"route '{route.name}' not found")
        self._ensure_embedded(route)
        self._routes[route.name] = route
        self.index.update(route)

    def get(self, name: str) -> Route | None:
        """
        Return a route by name.

        Algorithm:
            Look up the route in the internal mapping.

        Complexity:
            O(1).

        Args:
            name: Route name.

        Returns:
            The route when present, otherwise `None`.

        Example:
            >>> RouteLayer(
            ...     routes=[],
            ...     encoder=SentenceTransformerEncoder(show_progress=False),
            ... ).get("x") is None
            True
        """

        return self._routes.get(name)

    def list_routes(self) -> list[str]:
        """
        Return the registered route names.

        Algorithm:
            Materialize the internal route mapping keys.

        Complexity:
            O(R).

        Returns:
            Route names in insertion order.

        Example:
            >>> RouteLayer(
            ...     routes=[],
            ...     encoder=SentenceTransformerEncoder(show_progress=False),
            ... ).list_routes()
            []
        """

        return list(self._routes)

    def calibrate(
        self,
        test_queries: list[tuple[str, str | None]],
        metric: str = "f1",
    ) -> dict[str, float]:
        """
        Calibrate thresholds from labeled query data and apply the results.

        Algorithm:
            Delegate grid search to `ThresholdCalibrator`, then write the best
            global and per-route thresholds back into the layer configuration.

        Complexity:
            O(T × Q × R × U × D), where T is the number of thresholds explored.

        Args:
            test_queries: Labeled `(query, expected_route_name)` pairs.
            metric: Optimization metric.

        Returns:
            Mapping of route names to their calibrated thresholds.

        Example:
            >>> hasattr(RouteLayer, "calibrate")
            True
        """

        from semantic_router.calibration import ThresholdCalibrator

        calibrator = ThresholdCalibrator(self)
        result = calibrator.calibrate(test_queries, metric=metric, per_route=True)
        self.config.default_threshold = result.best_global_threshold
        for route_name, threshold in result.per_route_thresholds.items():
            route = self._routes.get(route_name)
            if route is not None:
                route.threshold = threshold
        return result.per_route_thresholds

    def suggest_threshold(self) -> float:
        """
        Suggest a threshold from the current route score distribution.

        Algorithm:
            Score each utterance against its own route and against the best other
            route, then choose the midpoint between the weaker positive tail and
            the strongest negative tail.

        Complexity:
            O(R² × U × D).

        Returns:
            Suggested global threshold.

        Example:
            >>> RouteLayer(
            ...     routes=[],
            ...     encoder=SentenceTransformerEncoder(show_progress=False),
            ... ).suggest_threshold()
            0.78
        """

        if not self._routes:
            return self.config.default_threshold
        positive_scores: list[float] = []
        negative_scores: list[float] = []
        route_items = list(self._routes.items())
        for route_name, route in route_items:
            for vector in route._utterance_vectors:
                positive_scores.append(
                    route.score(vector, strategy=self.config.routing_strategy)
                )
                other_scores = [
                    other.score(vector, strategy=self.config.routing_strategy)
                    for other_name, other in route_items
                    if other_name != route_name
                ]
                if other_scores:
                    negative_scores.append(max(other_scores))
        if not positive_scores or not negative_scores:
            return self.config.default_threshold
        suggested = (
            float(np.percentile(positive_scores, 10))
            + float(np.percentile(negative_scores, 95))
        ) / 2.0
        return float(
            np.clip(
                suggested,
                self.config.calibration_min_threshold,
                self.config.calibration_max_threshold,
            )
        )

    def save(self, path: str) -> None:
        """
        Serialize router state to JSON.

        Algorithm:
            Dump configuration, encoder metadata, and serialized routes to disk so
            a future process can restore routing state.

        Complexity:
            O(R × U × D).

        Args:
            path: Target JSON file path.

        Example:
            >>> hasattr(RouteLayer, "save")
            True
        """

        payload = {
            "config": self.config.to_dict(),
            "encoder": {"name": self.encoder.name},
            "routes": [route.to_dict() for route in self.routes],
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        """
        Restore router state from JSON.

        Algorithm:
            Deserialize saved config and routes, re-embed any route missing vector
            state, then rebuild the candidate index.

        Complexity:
            O(R × U × D) when re-embedding is needed.

        Args:
            path: JSON file path written by `save`.

        Example:
            >>> hasattr(RouteLayer, "load")
            True
        """

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.config = RouterConfig.from_dict(payload["config"])
        self._routes.clear()
        self.index.clear()
        for route_data in payload["routes"]:
            route = Route.from_dict(route_data)
            if not route.is_embedded:
                route.embed(self.encoder)
            self._routes[route.name] = route
        self._rebuild_index()
