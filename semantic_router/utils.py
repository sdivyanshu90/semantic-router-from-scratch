"""Pure numerical helpers used across the semantic router implementation."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
from tqdm.auto import tqdm


class SupportsEncode(Protocol):
    """Minimal protocol for objects that provide a batch `encode` method."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into an embedding matrix."""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two one-dimensional vectors.

    Algorithm:
        1. Validate that both vectors are one-dimensional.
        2. Compute their L2 norms.
        3. Divide the dot product by the norm product.
        4. Clamp the result to `[-1, 1]` to absorb floating-point drift.

    Complexity:
        O(D), where D is the vector dimensionality.

    Args:
        a: First vector with shape `(D,)`.
        b: Second vector with shape `(D,)`.

    Returns:
        Cosine similarity in the closed interval `[-1.0, 1.0]`.

    Raises:
        ValueError: If either vector is not one-dimensional, shapes differ, or a
            zero-norm vector is provided.

    Example:
        >>> cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        1.0
    """

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("cosine_similarity expects one-dimensional vectors")
    if a.shape != b.shape:
        raise ValueError("vector shapes must match")
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0.0 or b_norm == 0.0:
        raise ValueError("cosine similarity is undefined for zero-norm vectors")
    score = float(np.dot(a, b) / (a_norm * b_norm))
    return float(np.clip(score, -1.0, 1.0))


def cosine_similarity_matrix(queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """
    Compute a dense matrix of cosine similarities using vectorized NumPy.

    Algorithm:
        1. Validate two-dimensional input matrices.
        2. L2-normalize rows of both matrices.
        3. Compute the matrix product `queries @ keys.T` without Python loops.

    Complexity:
        O(Q × K × D), where Q is the number of queries, K is the number of keys,
        and D is the embedding dimensionality.

    Args:
        queries: Query matrix with shape `(Q, D)`.
        keys: Key matrix with shape `(K, D)`.

    Returns:
        Similarity matrix with shape `(Q, K)`.

    Raises:
        ValueError: If the inputs are not two-dimensional or dimensions differ.

    Example:
        >>> cosine_similarity_matrix(np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32)).shape
        (2, 2)
    """

    if queries.ndim != 2 or keys.ndim != 2:
        raise ValueError("cosine_similarity_matrix expects two-dimensional arrays")
    if queries.shape[1] != keys.shape[1]:
        raise ValueError("query and key matrices must share the same embedding dimension")
    normalized_queries = normalize(queries)
    normalized_keys = normalize(keys)
    return cast(np.ndarray, normalized_queries @ normalized_keys.T)


def batch_encode_progress(
    encoder: SupportsEncode,
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of texts in batches while optionally displaying progress.

    Algorithm:
        Slice the input into contiguous batches, encode each batch, and
        concatenate the resulting matrices along axis 0.

    Complexity:
        O(N × D) plus the cost of the underlying encoder, where N is the number
        of texts and D is the embedding dimensionality.

    Args:
        encoder: Encoder-like object exposing `encode(list[str])`.
        texts: Input texts to encode.
        batch_size: Number of texts per batch.
        show_progress: Whether to render a tqdm progress bar.

    Returns:
        Concatenated embedding matrix with shape `(N, D)`.

    Raises:
        ValueError: If `batch_size` is less than 1.

    Example:
        >>> class Dummy:
        ...     def encode(self, texts: list[str]) -> np.ndarray:
        ...         return np.ones((len(texts), 2), dtype=np.float32)
        >>> batch_encode_progress(Dummy(), ["a", "b"], batch_size=1, show_progress=False).shape
        (2, 2)
    """

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    batches: list[np.ndarray] = []
    iterator = range(0, len(texts), batch_size)
    progress = tqdm(iterator, disable=not show_progress, desc="Encoding batches")
    for start_index in progress:
        batch = texts[start_index : start_index + batch_size]
        batches.append(encoder.encode(batch))
    return cast(np.ndarray, np.concatenate(batches, axis=0))


def normalize(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors row-wise while preserving zero rows safely.

    Algorithm:
        1. Promote one-dimensional input to a single-row matrix.
        2. Compute row norms.
        3. Replace zeros with one to avoid division by zero.
        4. Divide each row by its safe norm.
        5. Restore the original dimensionality.

    Complexity:
        O(N × D), where N is the number of rows.

    Args:
        vectors: Either a vector of shape `(D,)` or matrix of shape `(N, D)`.

    Returns:
        Array with the same shape as the input and unit-length non-zero rows.

    Example:
        >>> normalize(np.array([[3.0, 4.0]], dtype=np.float32)).round(2).tolist()
        [[0.6, 0.8]]
    """

    was_one_dimensional = vectors.ndim == 1
    matrix = np.atleast_2d(vectors.astype(np.float32, copy=False))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = matrix / safe_norms
    if was_one_dimensional:
        return cast(np.ndarray, normalized[0])
    return cast(np.ndarray, normalized)


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Return indices of the top-k scores in descending order.

    Algorithm:
        Use `np.argpartition` to select the top-k candidates in linear time, then
        sort only that subset to obtain descending order.

    Complexity:
        O(N) for partitioning plus O(k log k) for sorting the selected subset,
        which is faster than a full O(N log N) sort when `k << N`.

    Args:
        scores: One-dimensional array of scores.
        k: Number of indices to return.

    Returns:
        Indices of the highest-scoring entries.

    Raises:
        ValueError: If `scores` is not one-dimensional or `k` is less than 1.

    Example:
        >>> top_k_indices(np.array([0.2, 0.9, 0.5]), 2).tolist()
        [1, 2]
    """

    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if k < 1:
        raise ValueError("k must be at least 1")
    if scores.size == 0:
        return np.array([], dtype=np.int64)
    effective_k = min(k, scores.size)
    partitioned = np.argpartition(scores, -effective_k)[-effective_k:]
    return cast(np.ndarray, partitioned[np.argsort(scores[partitioned])[::-1]])
