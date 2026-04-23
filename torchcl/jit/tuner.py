"""
Auto-Tuner — Queries the OpenCL device and selects optimal execution
parameters (workgroup sizes, tile sizes, vectorization) based on
the actual hardware capabilities.

This is what makes TorchCL work well on ANY chip — it adapts automatically.
"""

from __future__ import annotations

from torchcl.runtime.context import get_device_info


class AutoTuner:
    """Hardware-aware parameter selection for kernel execution."""

    def __init__(self) -> None:
        self._info: dict | None = None

    @property
    def info(self) -> dict:
        if self._info is None:
            self._info = get_device_info()
        return self._info

    def optimal_workgroup_1d(self, n: int) -> int:
        """Choose the best 1D workgroup size for n elements.

        Rules:
          - Never exceed device max_work_group_size
          - Use 256 for large arrays (good occupancy on most GPUs)
          - Use 128 for mid-size arrays
          - Use 64 for small arrays
          - Use n itself if very small
        """
        max_wg = self.info["max_work_group_size"]

        if n >= 65536:
            wg = min(256, max_wg)
        elif n >= 4096:
            wg = min(128, max_wg)
        elif n >= 256:
            wg = min(64, max_wg)
        else:
            wg = min(n, max_wg)

        # Ensure workgroup size is a power of 2
        wg = 1
        target = min(256, max_wg, n)
        while wg * 2 <= target:
            wg *= 2

        return max(wg, 1)

    def optimal_tile_size(self, M: int, N: int, K: int) -> int:
        """Choose the best tile size for matrix multiplication.

        Rules:
          - Use 16x16 tiles if local memory >= 8 KB (most GPUs)
          - Use 8x8 tiles if local memory is limited
          - Tiles must fit: 2 * TILE^2 * sizeof(float) <= local_mem
        """
        local_mem_kb = self.info["local_mem_size_kb"]
        max_wg = self.info["max_work_group_size"]

        if local_mem_kb >= 16 and max_wg >= 256 and min(M, N, K) >= 32:
            return 16
        elif local_mem_kb >= 4 and max_wg >= 64 and min(M, N, K) >= 16:
            return 8
        else:
            return 4

    def optimal_matmul_strategy(self, M: int, N: int, K: int) -> str:
        """Choose between naive and tiled matmul.

        Tiled is faster for large matrices, but the overhead of
        local memory management makes it slower for very small ones.
        """
        if min(M, N, K) >= 16 and self.info["local_mem_size_kb"] >= 4:
            return "tiled"
        return "naive"

    def should_vectorize(self) -> bool:
        """Check if the device benefits from explicit vectorization."""
        return self.info["preferred_vector_width_float"] > 1

    def max_alloc_elements(self, dtype_bytes: int = 4) -> int:
        """Maximum number of elements that can be allocated at once."""
        max_alloc_mb = self.info["max_mem_alloc_size_mb"]
        return (max_alloc_mb * 1024 * 1024) // dtype_bytes

    def summary(self) -> str:
        """Human-readable summary of tuning decisions."""
        info = self.info
        lines = [
            f"Device: {info['name']}",
            f"Compute Units: {info['max_compute_units']}",
            f"Max Workgroup: {info['max_work_group_size']}",
            f"Local Memory: {info['local_mem_size_kb']} KB",
            f"Global Memory: {info['global_mem_size_mb']} MB",
            f"Preferred 1D workgroup: {self.optimal_workgroup_1d(100000)}",
            f"Preferred matmul tile: {self.optimal_tile_size(512, 512, 512)}",
            f"Vectorize: {self.should_vectorize()}",
        ]
        return "\n".join(lines)


# ── Module-level singleton ───────────────────────────────────────────
_global_tuner: AutoTuner | None = None


def get_auto_tuner() -> AutoTuner:
    """Return the global auto-tuner."""
    global _global_tuner
    if _global_tuner is None:
        _global_tuner = AutoTuner()
    return _global_tuner
