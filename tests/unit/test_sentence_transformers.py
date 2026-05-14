"""Unit tests for `semantic_router.encoders.sentence_transformers`."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from semantic_router.encoders.sentence_transformers import SentenceTransformerEncoder


class DummySentenceTransformer:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        del batch_size, show_progress_bar, convert_to_numpy
        return np.asarray([[len(text), 1.0, 0.5] for text in texts], dtype=np.float32)


def test_sentence_transformer_encoder_lazy_load_and_encode(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=DummySentenceTransformer),
    )
    encoder = SentenceTransformerEncoder(show_progress=False)

    matrix = encoder.encode(["hello", "world"])

    assert matrix.shape == (2, 3)
    assert encoder.dimensions == 3
    assert encoder.name == "sentence-transformers:all-MiniLM-L6-v2"
