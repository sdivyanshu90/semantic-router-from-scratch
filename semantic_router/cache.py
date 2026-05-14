"""
Embedding cache implementations.

Cache flow:

Query ──► L1 LRU Cache ──► HIT  ──────────────────► return vector
              │
             MISS
              │
              ▼
         L2 Disk Cache ──► HIT  ──► populate L1 ──► return vector
              │
             MISS
              │
              ▼
         Encoder.encode() ──► populate L1 + L2 ──► return vector

The cache key scheme combines the encoder name and raw text before hashing with
SHA-256. The hash is used as the disk key, while the stored payload also keeps
the original `(text, encoder_name)` pair to guard against the extremely unlikely
event of a hash collision.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from diskcache import Cache


@dataclass(slots=True)
class CacheStats:
    """Lightweight snapshot of cache counters and storage size."""

    hits: int
    misses: int
    size_bytes: int


class EmbeddingCache:
    """
    Provide a two-level cache for embedding vectors.

    Algorithm:
        Keep hot vectors in an in-memory LRU map and persist a second copy to a
        disk-backed cache so repeated runs can avoid re-embedding unchanged text.

    Complexity:
        `get` and `set` are O(1) on average.

    Args:
        maxsize: Maximum number of vectors kept in the in-memory LRU cache.
        directory: Disk cache directory.

    Example:
        >>> cache = EmbeddingCache(maxsize=2)
        >>> cache.stats().hits
        0
    """

    def __init__(self, maxsize: int = 1024, directory: str | None = None) -> None:
        self.maxsize = maxsize
        self.directory = directory or os.path.join(
            Path.home(),
            ".semantic_router",
            "cache",
        )
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        self._l1: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._l2 = Cache(self.directory)
        self._hits = 0
        self._misses = 0

    def _hash_key(self, text: str, encoder_name: str) -> str:
        material = f"{encoder_name}\0{text}".encode()
        return hashlib.sha256(material).hexdigest()

    def _remember_l1(self, key: tuple[str, str], vector: np.ndarray) -> None:
        self._l1[key] = vector.astype(np.float32, copy=False)
        self._l1.move_to_end(key)
        while len(self._l1) > self.maxsize:
            self._l1.popitem(last=False)

    def get(self, text: str, encoder_name: str) -> np.ndarray | None:
        """
        Retrieve a cached vector from L1 or L2.

        Algorithm:
            Check the in-memory LRU first, then the disk cache, and promote disk
            hits back into L1.

        Complexity:
            O(1) on average.

        Args:
            text: Source text.
            encoder_name: Stable encoder identifier.

        Returns:
            Cached vector when present, otherwise `None`.

        Example:
            >>> EmbeddingCache(maxsize=1).get("hello", "demo") is None
            True
        """

        l1_key = (text, encoder_name)
        if l1_key in self._l1:
            self._hits += 1
            vector = self._l1[l1_key]
            self._l1.move_to_end(l1_key)
            return vector

        disk_key = self._hash_key(text, encoder_name)
        payload = self._l2.get(disk_key)
        if payload is None:
            self._misses += 1
            return None
        if payload["text"] != text or payload["encoder_name"] != encoder_name:
            self._misses += 1
            return None
        self._hits += 1
        vector = np.asarray(payload["vector"], dtype=np.float32)
        self._remember_l1(l1_key, vector)
        return vector

    def set(self, text: str, encoder_name: str, vector: np.ndarray) -> None:
        """
        Store a vector in both cache levels.

        Algorithm:
            Normalize storage by writing the vector to the in-memory LRU and the
            disk cache under the same collision-safe key.

        Complexity:
            O(1) on average.

        Args:
            text: Source text.
            encoder_name: Stable encoder identifier.
            vector: Embedding vector to cache.

        Example:
            >>> cache = EmbeddingCache(maxsize=1)
            >>> cache.set("hello", "demo", np.array([1.0], dtype=np.float32))
        """

        normalized = vector.astype(np.float32, copy=False)
        l1_key = (text, encoder_name)
        self._remember_l1(l1_key, normalized)
        self._l2.set(
            self._hash_key(text, encoder_name),
            {"text": text, "encoder_name": encoder_name, "vector": normalized},
        )

    def clear(self, level: str = "all") -> None:
        """
        Clear cache state at the requested level.

        Algorithm:
            Empty the in-memory map, the disk cache, or both depending on the
            selected level.

        Complexity:
            O(1) for L1, O(N) for L2 clear depending on backend size.

        Args:
            level: One of `"l1"`, `"l2"`, or `"all"`.

        Raises:
            ValueError: If an unsupported level is requested.

        Example:
            >>> cache = EmbeddingCache(maxsize=1)
            >>> cache.clear("l1")
        """

        if level not in {"l1", "l2", "all"}:
            raise ValueError("level must be one of 'l1', 'l2', or 'all'")
        if level in {"l1", "all"}:
            self._l1.clear()
        if level in {"l2", "all"}:
            self._l2.clear()

    def stats(self) -> CacheStats:
        """
        Return current cache hit, miss, and size counters.

        Algorithm:
            Sum the approximate in-memory footprint with the disk cache volume.

        Complexity:
            O(L1) for in-memory size accounting.

        Returns:
            `CacheStats` snapshot.

        Example:
            >>> isinstance(EmbeddingCache(maxsize=1).stats().size_bytes, int)
            True
        """

        l1_bytes = sum(vector.nbytes for vector in self._l1.values())
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            size_bytes=l1_bytes + int(self._l2.volume()),
        )
