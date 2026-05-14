"""Unit tests for `semantic_router.utils`."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_router.encoders.base import BaseEncoder
from semantic_router.utils import (
    batch_encode_progress,
    cosine_similarity,
    cosine_similarity_matrix,
    normalize,
    top_k_indices,
)


class DummyEncoder(BaseEncoder):
    def encode(self, texts: list[str]) -> np.ndarray:
        return normalize(np.ones((len(texts), 3), dtype=np.float32))

    async def async_encode(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    @property
    def dimensions(self) -> int:
        return 3

    @property
    def name(self) -> str:
        return "dummy"


def test_cosine_similarity_rejects_zero_norm() -> None:
    with pytest.raises(ValueError):
        cosine_similarity(
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        )


def test_cosine_similarity_matrix_returns_expected_shape() -> None:
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    keys = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    matrix = cosine_similarity_matrix(queries, keys)

    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(1.0)


def test_batch_encode_progress_concatenates_batches() -> None:
    matrix = batch_encode_progress(
        DummyEncoder(),
        ["a", "b", "c"],
        batch_size=2,
        show_progress=False,
    )

    assert matrix.shape == (3, 3)


def test_normalize_preserves_zero_rows() -> None:
    matrix = normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))

    assert matrix[0, 0] == pytest.approx(0.6)
    assert np.allclose(matrix[1], np.array([0.0, 0.0], dtype=np.float32))


def test_top_k_indices_returns_descending_order() -> None:
    indices = top_k_indices(np.array([0.1, 0.9, 0.5, 0.7], dtype=np.float32), 3)

    assert indices.tolist() == [1, 3, 2]
