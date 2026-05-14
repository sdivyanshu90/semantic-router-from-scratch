"""Additional coverage-oriented tests for `semantic_router.config`."""

from __future__ import annotations

import pytest

from semantic_router.config import RouterConfig
from semantic_router.exceptions import RouteConfigurationError


def test_config_rejects_invalid_top_k() -> None:
    with pytest.raises(RouteConfigurationError):
        RouterConfig(top_k=0)


def test_config_rejects_invalid_batch_size() -> None:
    with pytest.raises(RouteConfigurationError):
        RouterConfig(default_batch_size=0)


def test_config_rejects_invalid_calibration_bounds() -> None:
    with pytest.raises(RouteConfigurationError):
        RouterConfig(calibration_min_threshold=0.9, calibration_max_threshold=0.8)


def test_config_rejects_invalid_calibration_steps() -> None:
    with pytest.raises(RouteConfigurationError):
        RouterConfig(calibration_steps=1)
