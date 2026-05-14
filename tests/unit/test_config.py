"""Unit tests for `semantic_router.config`."""

from __future__ import annotations

import pytest

from semantic_router.config import RouterConfig
from semantic_router.exceptions import RouteConfigurationError


def test_config_round_trip() -> None:
    config = RouterConfig(default_threshold=0.81, top_k=3)

    restored = RouterConfig.from_dict(config.to_dict())

    assert restored == config


def test_config_rejects_invalid_threshold() -> None:
    with pytest.raises(RouteConfigurationError):
        RouterConfig(default_threshold=1.5)
