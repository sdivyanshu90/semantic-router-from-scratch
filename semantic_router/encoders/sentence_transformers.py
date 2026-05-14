"""
Sentence-Transformers encoder backend.

Recommended models:

┌─────────────────────────────┬────────┬──────────┬───────────────┐
│ Model                       │  Dims  │  Speed   │  Quality      │
├─────────────────────────────┼────────┼──────────┼───────────────┤
│ all-MiniLM-L6-v2 (default) │  384   │  Fast    │  Good         │
│ all-mpnet-base-v2           │  768   │  Medium  │  Better       │
│ paraphrase-multilingual-*   │  768   │  Medium  │  Multilingual │
│ BAAI/bge-large-en-v1.5      │ 1024   │  Slow    │  Best         │
└─────────────────────────────┴────────┴──────────┴───────────────┘
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import numpy as np

from semantic_router.encoders.base import BaseEncoder
from semantic_router.exceptions import EncoderError
from semantic_router.utils import normalize


class SentenceTransformerEncoder(BaseEncoder):
    """
    Encode text locally with a lazily loaded Sentence-Transformers model.

    Algorithm:
        1. Lazily load the requested model on first use under a thread lock.
        2. Resolve cached vectors when a cache backend is attached.
        3. Batch-encode only uncached texts.
        4. L2-normalize every returned vector.

    Complexity:
        O(N × D) plus the underlying transformer inference cost.

    Args:
        model_name: Sentence-Transformers model identifier.
        batch_size: Number of texts per inference batch.
        show_progress: Whether to show the encoder progress bar.
        cache: Optional embedding cache.

    Example:
        >>> encoder = SentenceTransformerEncoder(model_name="all-MiniLM-L6-v2", show_progress=False)
        >>> encoder.name.startswith("sentence-transformers:")
        True
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        show_progress: bool = True,
        cache: object | None = None,
    ) -> None:
        super().__init__(cache=cache)
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress = show_progress
        self._model: Any | None = None
        self._dimensions: int | None = None
        self._load_lock = threading.Lock()

    def _ensure_model_loaded(self) -> Any:
        """
        Lazily load the Sentence-Transformers model exactly once.

        Algorithm:
            Use double-checked locking so concurrent callers share one model
            initialization path without serializing every inference call.

        Complexity:
            O(1) after the model is loaded; initial load cost depends on model size.

        Returns:
            Loaded `SentenceTransformer` instance.

        Raises:
            EncoderError: If the dependency is missing or the model cannot load.

        Example:
            >>> encoder = SentenceTransformerEncoder(show_progress=False)
            >>> encoder._model is None
            True
        """

        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:  # pragma: no cover - import guard
                    raise EncoderError(
                        "sentence-transformers is required for SentenceTransformerEncoder"
                    ) from exc
                try:
                    self._model = SentenceTransformer(self.model_name)
                    self._dimensions = int(self._model.get_sentence_embedding_dimension())
                except Exception as exc:  # pragma: no cover - backend failure
                    raise EncoderError(f"failed to load model '{self.model_name}'") from exc
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a batch of texts into a normalized embedding matrix.

        Algorithm:
            Look up cached vectors first, encode only missing texts with the local
            transformer model, then normalize and merge all results in the
            original input order.

        Complexity:
            O(N × D) plus local model inference cost.

        Args:
            texts: Input texts to encode.

        Returns:
            Float32 matrix with shape `(N, D)`.

        Example:
            >>> encoder = SentenceTransformerEncoder(show_progress=False)
            >>> encoder.dimensions > 0
            True
        """

        if not texts:
            dimensions = self._dimensions or 0
            return np.zeros((0, dimensions), dtype=np.float32)

        cached_vectors: dict[int, np.ndarray] = {}
        missing_texts: list[str] = []
        missing_indices: list[int] = []

        if self.cache is not None:
            for index, text in enumerate(texts):
                cached = self.cache.get(text, self.name)
                if cached is None:
                    missing_texts.append(text)
                    missing_indices.append(index)
                else:
                    cached_vectors[index] = cached.astype(np.float32, copy=False)
        else:
            missing_texts = list(texts)
            missing_indices = list(range(len(texts)))

        if missing_texts:
            model = self._ensure_model_loaded()
            encoded = model.encode(
                missing_texts,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress,
                convert_to_numpy=True,
            )
            normalized = normalize(np.asarray(encoded, dtype=np.float32))
            for index, vector in zip(missing_indices, normalized, strict=True):
                cached_vectors[index] = vector
                if self.cache is not None:
                    self.cache.set(texts[index], self.name, vector)

        ordered = [cached_vectors[index] for index in range(len(texts))]
        return np.stack(ordered).astype(np.float32, copy=False)

    async def async_encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode texts asynchronously by delegating to a worker thread.

        Algorithm:
            Offload synchronous local inference to `asyncio.to_thread` so async
            applications do not block the event loop.

        Complexity:
            Matches `encode` plus thread scheduling overhead.

        Args:
            texts: Input texts to encode.

        Returns:
            Float32 normalized embedding matrix.

        Example:
            >>> encoder = SentenceTransformerEncoder(show_progress=False)
            >>> hasattr(encoder, "async_encode")
            True
        """

        return await asyncio.to_thread(self.encode, texts)

    @property
    def dimensions(self) -> int:
        """
        Return the dimensionality of embeddings produced by the model.

        Algorithm:
            Use the cached dimension value when available; otherwise load the
            model and ask it for the sentence embedding size.

        Complexity:
            O(1) after model load.

        Returns:
            Embedding dimensionality.

        Example:
            >>> isinstance(SentenceTransformerEncoder(show_progress=False).model_name, str)
            True
        """

        if self._dimensions is None:
            self._ensure_model_loaded()
        if self._dimensions is None:  # pragma: no cover - defensive check
            raise EncoderError("model dimensions are unavailable")
        return self._dimensions

    @property
    def name(self) -> str:
        """
        Return a stable backend name suitable for logs and cache keys.

        Algorithm:
            Prefix the model identifier with the backend family name.

        Complexity:
            O(1).

        Returns:
            Stable backend identifier.

        Example:
            >>> SentenceTransformerEncoder(show_progress=False).name
            'sentence-transformers:all-MiniLM-L6-v2'
        """

        return f"sentence-transformers:{self.model_name}"
