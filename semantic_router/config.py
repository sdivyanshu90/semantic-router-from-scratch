"""Configuration objects and defaults for semantic routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, cast

from semantic_router.exceptions import RouteConfigurationError

RoutingStrategy = Literal["centroid", "max", "mean"]


@dataclass(slots=True)
class RouterConfig:
    """
    Store operational settings for a `RouteLayer`.

    Algorithm:
        The configuration object centralizes routing constants so the router can
        resolve thresholds, top-k retrieval, and scoring strategy without hidden
        magic numbers spread throughout the codebase.

    Complexity:
        Construction and validation are O(1).

    Args:
        default_threshold: Fallback similarity threshold used when a route does not
            provide its own override.
        top_k: Number of candidate routes retrieved from the index before exact
            scoring.
        routing_strategy: Similarity aggregation strategy used by routes.
        default_batch_size: Default batch size for bulk encoding operations.
        calibration_min_threshold: Lower bound for calibration grid search.
        calibration_max_threshold: Upper bound for calibration grid search.
        calibration_steps: Number of threshold steps evaluated during calibration.
        cache_enabled: Whether encoders should use cache-aware code paths when a
            cache backend is attached.

    Raises:
        RouteConfigurationError: If any configuration value is outside the valid
            range.

    Example:
        >>> config = RouterConfig(default_threshold=0.8, top_k=3)
        >>> assert config.resolve_threshold(None) == 0.8
    """

    default_threshold: float = 0.78
    top_k: int = 5
    routing_strategy: RoutingStrategy = "centroid"
    default_batch_size: int = 32
    calibration_min_threshold: float = 0.10
    calibration_max_threshold: float = 0.99
    calibration_steps: int = 90
    cache_enabled: bool = True

    def __post_init__(self) -> None:
        """
        Validate configuration invariants after dataclass construction.

        Algorithm:
            Perform a small set of range checks on each field to reject invalid
            router state early.

        Complexity:
            O(1).

        Raises:
            RouteConfigurationError: If any field violates its allowed range.

        Example:
            >>> RouterConfig(default_threshold=0.7, top_k=2)
            RouterConfig(
                default_threshold=0.7,
                top_k=2,
                routing_strategy='centroid',
                default_batch_size=32,
                calibration_min_threshold=0.1,
                calibration_max_threshold=0.99,
                calibration_steps=90,
                cache_enabled=True,
            )
        """

        if not 0.0 <= self.default_threshold <= 1.0:
            raise RouteConfigurationError("default_threshold must be between 0.0 and 1.0")
        if self.top_k < 1:
            raise RouteConfigurationError("top_k must be at least 1")
        if self.routing_strategy not in {"centroid", "max", "mean"}:
            raise RouteConfigurationError(
                "routing_strategy must be one of 'centroid', 'max', or 'mean'"
            )
        if self.default_batch_size < 1:
            raise RouteConfigurationError("default_batch_size must be at least 1")
        bounds_are_valid = (
            0.0
            <= self.calibration_min_threshold
            < self.calibration_max_threshold
            <= 1.0
        )
        if not bounds_are_valid:
            raise RouteConfigurationError(
                "calibration threshold bounds must satisfy 0.0 <= min < max <= 1.0"
            )
        if self.calibration_steps < 2:
            raise RouteConfigurationError("calibration_steps must be at least 2")

    def resolve_threshold(self, route_threshold: float | None) -> float:
        """
        Resolve the effective threshold for a route.

        Algorithm:
            Return the route-specific threshold when provided, otherwise fall back
            to the router default threshold.

        Complexity:
            O(1).

        Args:
            route_threshold: Optional route-level override.

        Returns:
            Effective similarity threshold.

        Example:
            >>> RouterConfig(default_threshold=0.81).resolve_threshold(None)
            0.81
        """

        return self.default_threshold if route_threshold is None else route_threshold

    def to_dict(self) -> dict[str, object]:
        """
        Serialize the configuration to a plain dictionary.

        Algorithm:
            Delegate to `dataclasses.asdict` so the result is JSON-serializable.

        Complexity:
            O(1).

        Returns:
            Dictionary representation of the configuration.

        Example:
            >>> RouterConfig().to_dict()["top_k"]
            5
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RouterConfig:
        """
        Build a configuration instance from serialized data.

        Algorithm:
            Expand the input mapping as dataclass constructor arguments.

        Complexity:
            O(1).

        Args:
            data: Serialized configuration values.

        Returns:
            Reconstructed `RouterConfig` instance.

        Example:
            >>> RouterConfig.from_dict({"default_threshold": 0.8, "top_k": 4})
            RouterConfig(
                default_threshold=0.8,
                top_k=4,
                routing_strategy='centroid',
                default_batch_size=32,
                calibration_min_threshold=0.1,
                calibration_max_threshold=0.99,
                calibration_steps=90,
                cache_enabled=True,
            )
        """

        return cls(
            default_threshold=cast(float, data.get("default_threshold", 0.78)),
            top_k=cast(int, data.get("top_k", 5)),
            routing_strategy=cast(
                RoutingStrategy,
                data.get("routing_strategy", "centroid"),
            ),
            default_batch_size=cast(int, data.get("default_batch_size", 32)),
            calibration_min_threshold=cast(
                float,
                data.get("calibration_min_threshold", 0.10),
            ),
            calibration_max_threshold=cast(
                float,
                data.get("calibration_max_threshold", 0.99),
            ),
            calibration_steps=cast(int, data.get("calibration_steps", 90)),
            cache_enabled=cast(bool, data.get("cache_enabled", True)),
        )
