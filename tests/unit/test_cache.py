"""Unit tests for `semantic_router.cache`."""

from __future__ import annotations

import numpy as np

from semantic_router.cache import EmbeddingCache


def test_cache_miss_then_hit(tmp_path) -> None:
    cache = EmbeddingCache(maxsize=2, directory=str(tmp_path / "cache"))
    vector = np.array([1.0, 0.0], dtype=np.float32)

    assert cache.get("hello", "demo") is None
    cache.set("hello", "demo", vector)
    cached = cache.get("hello", "demo")

    assert cached is not None
    assert np.allclose(cached, vector)


def test_clear_l1_only_preserves_l2(tmp_path) -> None:
    cache = EmbeddingCache(maxsize=2, directory=str(tmp_path / "cache"))
    vector = np.array([1.0, 2.0], dtype=np.float32)
    cache.set("hello", "demo", vector)
    assert cache.get("hello", "demo") is not None

    cache.clear("l1")
    cached = cache.get("hello", "demo")

    assert cached is not None
    assert np.allclose(cached, vector)


def test_cache_stats_report_hits_and_misses(tmp_path) -> None:
    cache = EmbeddingCache(maxsize=2, directory=str(tmp_path / "cache"))
    cache.get("miss", "demo")
    cache.set("hit", "demo", np.array([1.0], dtype=np.float32))
    cache.get("hit", "demo")
    stats = cache.stats()

    assert stats.misses == 1
    assert stats.hits == 1
    assert stats.size_bytes >= 1
