"""
Kernel Cache — In-memory LRU cache for compiled OpenCL kernels.

Avoids recompiling the same kernel source when the same operation
is called repeatedly (e.g., during a training loop).
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

import pyopencl as cl


class KernelCache:
    """Thread-safe LRU cache for compiled OpenCL kernels.

    Args:
        max_size: Maximum number of cached kernels. Oldest entries
                  are evicted when the cache is full.
    """

    def __init__(self, max_size: int = 512) -> None:
        self._cache: OrderedDict[str, cl.Kernel] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> cl.Kernel | None:
        """Look up a cached kernel. Returns None on miss."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: str, kernel: cl.Kernel) -> None:
        """Insert or update a cached kernel."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)  # Evict oldest
            self._cache[key] = kernel

    def clear(self) -> None:
        """Clear all cached kernels."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{self._hits / total:.1%}" if total > 0 else "N/A",
            }

    @staticmethod
    def make_key(*parts) -> str:
        """Create a cache key from multiple parts (hashed if too long)."""
        raw = ":".join(str(p) for p in parts)
        if len(raw) > 128:
            return hashlib.sha256(raw.encode()).hexdigest()
        return raw


# ── Module-level singleton ───────────────────────────────────────────
_global_cache: KernelCache | None = None


def get_kernel_cache() -> KernelCache:
    """Return the global kernel cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = KernelCache()
    return _global_cache
