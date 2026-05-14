"""Threshold calibration utilities for semantic routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from semantic_router.exceptions import CalibrationError

if TYPE_CHECKING:
    from semantic_router.layer import RouteLayer


@dataclass(slots=True)
class CalibrationResult:
    """
    Hold the outcome of threshold calibration.

    Algorithm:
        Store the best global threshold, optional per-route thresholds, the full
        metric curve, and a confusion matrix evaluated at the chosen threshold.

    Complexity:
        Construction is O(1).

    Example:
        >>> result = CalibrationResult(
        ...     0.5,
        ...     {},
        ...     pd.DataFrame(),
        ...     np.zeros((1, 1), dtype=int),
        ... )
        >>> result.best_global_threshold
        0.5
    """

    best_global_threshold: float
    per_route_thresholds: dict[str, float]
    calibration_curve: Any
    confusion_matrix: np.ndarray

    def plot(self) -> None:
        """
        Render a calibration curve plot with precision, recall, and F1 lines.

        Algorithm:
            Plot the metric columns in the stored DataFrame against threshold.

        Complexity:
            O(T), where T is the number of threshold points.

        Example:
            >>> hasattr(CalibrationResult, "plot")
            True
        """

        import matplotlib.pyplot as plt

        if self.calibration_curve.empty:
            raise CalibrationError("cannot plot an empty calibration curve")
        axis = self.calibration_curve.plot(
            x="threshold",
            y=["precision", "recall", "f1", "accuracy"],
            figsize=(10, 6),
            title="Calibration Curve",
        )
        axis.axvline(
            self.best_global_threshold,
            color="black",
            linestyle="--",
            label="best",
        )
        axis.legend()
        plt.tight_layout()
        plt.show()

    def summary(self) -> str:
        """
        Format per-route thresholds as an ASCII table.

        Algorithm:
            Compute the widest route label, then render aligned rows for every
            route threshold entry.

        Complexity:
            O(R), where R is the number of routes.

        Returns:
            Human-readable ASCII table.

        Example:
            >>> result = CalibrationResult(
            ...     0.5,
            ...     {"x": 0.6},
            ...     pd.DataFrame(),
            ...     np.zeros((1, 1), dtype=int),
            ... )
            >>> result.summary().splitlines()[0]
            'Per-route thresholds'
        """

        if not self.per_route_thresholds:
            return "Per-route thresholds\n(no calibrated per-route overrides)"
        width = max(len(name) for name in self.per_route_thresholds)
        lines = [
            "Per-route thresholds",
            f"{'Route'.ljust(width)} | Threshold",
            f"{'-' * width}-+----------",
        ]
        for name, threshold in sorted(self.per_route_thresholds.items()):
            lines.append(f"{name.ljust(width)} | {threshold:.3f}")
        return "\n".join(lines)


class ThresholdCalibrator:
    """
    Grid-search thresholds for one semantic router.

    Algorithm:
        Pre-compute route scores for all labeled queries, then sweep threshold
        values and compute the requested metric at every step.

    Complexity:
        O(T × Q × R × U × D).

    Args:
        route_layer: Route layer being calibrated.

    Example:
        >>> ThresholdCalibrator.__name__
        'ThresholdCalibrator'
    """

    def __init__(self, route_layer: RouteLayer) -> None:
        self.route_layer = route_layer

    @staticmethod
    def _metric_from_counts(
        tp: int,
        fp: int,
        fn: int,
        correct: int,
        total: int,
    ) -> dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        accuracy = correct / total if total else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    def _predict_name(self, scores: dict[str, float], threshold: float) -> str | None:
        if not scores:
            return None
        best_name, best_score = max(scores.items(), key=lambda item: item[1])
        return best_name if best_score >= threshold else None

    def _confusion_matrix(
        self,
        labels: list[str | None],
        expected: list[str | None],
        predicted: list[str | None],
    ) -> np.ndarray:
        index = {label: position for position, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for gold, guess in zip(expected, predicted, strict=True):
            matrix[index[gold], index[guess]] += 1
        return matrix

    def calibrate(
        self,
        labeled_queries: list[tuple[str, str | None]],
        thresholds: list[float] | None = None,
        metric: str = "f1",
        per_route: bool = True,
    ) -> CalibrationResult:
        """
        Grid-search threshold values and return the best-performing result.

        Algorithm:
            1. Score every labeled query against every route.
            2. Sweep the threshold grid.
            3. Compute precision, recall, F1, and accuracy at each threshold.
            4. Select the threshold that maximizes the requested metric.
            5. Optionally repeat a one-vs-rest search per route.

        Complexity:
            O(T × Q × R × U × D).

        Args:
            labeled_queries: `(query, expected_route_name_or_None)` pairs.
            thresholds: Optional threshold grid.
            metric: Optimization metric.
            per_route: Whether to compute one-vs-rest per-route thresholds.

        Returns:
            `CalibrationResult` containing best thresholds and calibration data.

        Raises:
            CalibrationError: If inputs are empty or metric is unsupported.

        Example:
            >>> hasattr(ThresholdCalibrator, "calibrate")
            True
        """

        if not labeled_queries:
            raise CalibrationError("labeled_queries must not be empty")
        valid_metrics = {"f1", "precision", "recall", "accuracy"}
        if metric not in valid_metrics:
            raise CalibrationError(f"metric must be one of {sorted(valid_metrics)}")

        threshold_grid = thresholds or np.linspace(
            self.route_layer.config.calibration_min_threshold,
            self.route_layer.config.calibration_max_threshold,
            self.route_layer.config.calibration_steps,
        ).tolist()

        queries = [query for query, _expected in labeled_queries]
        expected = [expected_name for _query, expected_name in labeled_queries]
        query_vectors = self.route_layer.encoder.encode(queries)
        route_names = self.route_layer.list_routes()
        query_scores = [
            self.route_layer._score_routes_for_vector(vector)
            for vector in query_vectors
        ]

        rows: list[dict[str, float]] = []
        best_threshold = threshold_grid[0]
        best_metric_value = -1.0
        best_predictions: list[str | None] = []
        for threshold in threshold_grid:
            predictions = [self._predict_name(scores, threshold) for scores in query_scores]
            tp = sum(
                guess == gold and gold is not None
                for gold, guess in zip(expected, predictions, strict=True)
            )
            fp = sum(
                guess is not None and guess != gold
                for gold, guess in zip(expected, predictions, strict=True)
            )
            fn = sum(
                gold is not None and guess != gold
                for gold, guess in zip(expected, predictions, strict=True)
            )
            correct = sum(
                gold == guess for gold, guess in zip(expected, predictions, strict=True)
            )
            metrics = self._metric_from_counts(tp, fp, fn, correct, len(labeled_queries))
            rows.append({"threshold": float(threshold), **metrics})
            if metrics[metric] > best_metric_value:
                best_metric_value = metrics[metric]
                best_threshold = float(threshold)
                best_predictions = predictions

        per_route_thresholds: dict[str, float] = {}
        if per_route:
            for route_name in route_names:
                route_best_threshold = threshold_grid[0]
                route_best_metric_value = -1.0
                for threshold in threshold_grid:
                    tp = fp = fn = correct = 0
                    for scores, expected_name in zip(query_scores, expected, strict=True):
                        predicted_positive = scores.get(route_name, -1.0) >= threshold
                        actual_positive = expected_name == route_name
                        if predicted_positive and actual_positive:
                            tp += 1
                        elif predicted_positive and not actual_positive:
                            fp += 1
                        elif not predicted_positive and actual_positive:
                            fn += 1
                        else:
                            correct += 1
                    metrics = self._metric_from_counts(
                        tp,
                        fp,
                        fn,
                        correct,
                        len(labeled_queries),
                    )
                    if metrics[metric] > route_best_metric_value:
                        route_best_metric_value = metrics[metric]
                        route_best_threshold = float(threshold)
                per_route_thresholds[route_name] = route_best_threshold

        labels = [*route_names, None]
        confusion = self._confusion_matrix(labels, expected, best_predictions)
        return CalibrationResult(
            best_global_threshold=best_threshold,
            per_route_thresholds=per_route_thresholds,
            calibration_curve=pd.DataFrame(rows),
            confusion_matrix=confusion,
        )
