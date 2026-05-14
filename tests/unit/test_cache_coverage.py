"""Additional coverage-oriented tests for `semantic_router.cache`."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_router.cache import EmbeddingCache


def test_cache_clear_rejects_invalid_level(tmp_path) -> None:
    cache = EmbeddingCache(maxsize=2, directory=str(tmp_path / "cache"))

    with pytest.raises(ValueError):
        cache.clear("bad-level")


def test_cache_disk_payload_mismatch_counts_as_miss(tmp_path) -> None:
    cache = EmbeddingCache(maxsize=2, directory=str(tmp_path / "cache"))
    disk_key = cache._hash_key("hello", "demo")
    cache._l2.set(
        disk_key,
        {
            "text": "other",
            "encoder_name": "different",
            "vector": np.array([1.0], dtype=np.float32),
        },
    )

    assert cache.get("hello", "demo") is None
    assert cache.stats().misses == 1
