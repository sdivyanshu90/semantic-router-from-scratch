"""Additional coverage-oriented tests for `semantic_router.calibration`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from semantic_router.calibration import CalibrationResult, ThresholdCalibrator
from semantic_router.exceptions import CalibrationError


def test_calibration_plot_raises_for_empty_curve() -> None:
    result = CalibrationResult(
        best_global_threshold=0.5,
        per_route_thresholds={},
        calibration_curve=pd.DataFrame(),
        confusion_matrix=np.zeros((1, 1), dtype=np.int64),
    )

    with pytest.raises(CalibrationError):
        result.plot()


def test_calibration_plot_renders_for_populated_curve(monkeypatch) -> None:
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    result = CalibrationResult(
        best_global_threshold=0.6,
        per_route_thresholds={"travel": 0.6},
        calibration_curve=pd.DataFrame(
            [
                {"threshold": 0.5, "precision": 0.8, "recall": 0.7, "f1": 0.75, "accuracy": 0.8},
                {"threshold": 0.6, "precision": 0.9, "recall": 0.7, "f1": 0.79, "accuracy": 0.82},
            ]
        ),
        confusion_matrix=np.zeros((2, 2), dtype=np.int64),
    )

    result.plot()


def test_calibration_summary_without_route_thresholds() -> None:
    result = CalibrationResult(
        best_global_threshold=0.5,
        per_route_thresholds={},
        calibration_curve=pd.DataFrame(),
        confusion_matrix=np.zeros((1, 1), dtype=np.int64),
    )

    assert "no calibrated per-route overrides" in result.summary()


def test_calibrate_rejects_empty_queries(embedded_layer) -> None:
    with pytest.raises(CalibrationError):
        ThresholdCalibrator(embedded_layer).calibrate([])


def test_calibrate_rejects_invalid_metric(embedded_layer) -> None:
    with pytest.raises(CalibrationError):
        ThresholdCalibrator(embedded_layer).calibrate(
            [("book a flight", "travel")],
            metric="bad-metric",
        )


def test_calibrate_can_skip_per_route_thresholds(embedded_layer) -> None:
    result = ThresholdCalibrator(embedded_layer).calibrate(
        [("book a flight", "travel"), ("unknown blorb", None)],
        thresholds=[0.5, 0.7],
        per_route=False,
    )

    assert result.per_route_thresholds == {}
