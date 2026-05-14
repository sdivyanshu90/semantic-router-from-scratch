"""Unit tests for `semantic_router.encoders.base`."""

from __future__ import annotations

import numpy as np

from semantic_router.encoders.base import BaseEncoder
from semantic_router.utils import normalize


class UnitEncoder(BaseEncoder):
    def encode(self, texts: list[str]) -> np.ndarray:
        return normalize(np.ones((len(texts), 2), dtype=np.float32))

    async def async_encode(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts)

    @property
    def dimensions(self) -> int:
        return 2

    @property
    def name(self) -> str:
        return "unit"


def test_encode_single_returns_vector() -> None:
    vector = UnitEncoder().encode_single("hello")

    assert vector.shape == (2,)
