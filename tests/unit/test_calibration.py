"""Unit tests for `semantic_router.calibration`."""

from __future__ import annotations

from semantic_router.calibration import ThresholdCalibrator
from semantic_router.layer import RouteLayer


def test_calibration_returns_thresholds_and_curve(embedded_layer: RouteLayer) -> None:
    calibrator = ThresholdCalibrator(embedded_layer)
    labeled_queries = [
        ("book a flight to Paris", "travel"),
        ("will it rain tomorrow", "weather"),
        ("play some jazz music", "music"),
        ("check my bank balance", "finance"),
        ("nonsense blorb phrase", None),
    ]

    result = calibrator.calibrate(labeled_queries, thresholds=[0.5, 0.7, 0.9])

    assert result.best_global_threshold in {0.5, 0.7, 0.9}
    assert set(result.per_route_thresholds) == set(embedded_layer.list_routes())
    assert not result.calibration_curve.empty
    assert result.confusion_matrix.shape[0] == len(embedded_layer.list_routes()) + 1


def test_calibration_summary_contains_route_names(embedded_layer: RouteLayer) -> None:
    result = ThresholdCalibrator(embedded_layer).calibrate(
        [("book a flight", "travel"), ("blorb blorb", None)],
        thresholds=[0.5, 0.8],
    )

    summary = result.summary()

    assert "Per-route thresholds" in summary
    assert "travel" in summary
