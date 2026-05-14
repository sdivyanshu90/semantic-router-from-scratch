"""Additional coverage-oriented tests for `semantic_router.utils`."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_router.utils import (
    batch_encode_progress,
    cosine_similarity,
    cosine_similarity_matrix,
    top_k_indices,
)


class TinyEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


def test_cosine_similarity_rejects_bad_dimensions() -> None:
    with pytest.raises(ValueError):
        cosine_similarity(
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        )


def test_cosine_similarity_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        cosine_similarity(
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )


def test_cosine_similarity_matrix_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        cosine_similarity_matrix(
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
        )

    with pytest.raises(ValueError):
        cosine_similarity_matrix(
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        )


def test_batch_encode_progress_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError):
        batch_encode_progress(TinyEncoder(), ["a"], batch_size=0, show_progress=False)


def test_top_k_indices_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        top_k_indices(np.array([[1.0, 2.0]], dtype=np.float32), 1)

    with pytest.raises(ValueError):
        top_k_indices(np.array([1.0, 2.0], dtype=np.float32), 0)

    assert top_k_indices(np.array([], dtype=np.float32), 1).size == 0
