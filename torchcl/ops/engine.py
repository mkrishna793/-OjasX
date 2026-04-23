"""
OpenCL Compute Engine — The central execution layer that bridges PyTorch
tensors to OpenCL kernels.

This module:
  1. Converts PyTorch tensors ↔ OpenCL buffers (via numpy)
  2. Launches compiled OpenCL kernels with correct workgroup sizes
  3. Returns results as PyTorch tensors

All operator modules (basic, matrix, activation, reduction) delegate here.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pyopencl as cl
import torch

from torchcl.runtime.context import get_queue, get_device_info, synchronize
from torchcl.runtime.memory import CLBuffer, get_buffer_pool
from torchcl.kernels.registry import get_kernel_registry

# ── Dtype mapping ────────────────────────────────────────────────────
_TORCH_TO_NP = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
}

_NP_TO_TORCH = {v: k for k, v in _TORCH_TO_NP.items()}


class OpenCLEngine:
    """Central compute engine — converts tensors, launches kernels, returns results."""

    def __init__(self) -> None:
        self._pool = get_buffer_pool()
        self._registry = get_kernel_registry()

    # ── Tensor ↔ Buffer conversion ───────────────────────────────

    def tensor_to_buffer(self, tensor: torch.Tensor) -> CLBuffer:
        """Upload a PyTorch (CPU) tensor to an OpenCL buffer."""
        np_array = tensor.detach().cpu().numpy()
        return self._pool.host_to_device(np_array)

    def buffer_to_tensor(
        self,
        cl_buf: CLBuffer,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Download an OpenCL buffer back to a PyTorch CPU tensor."""
        np_dtype = _TORCH_TO_NP.get(dtype, np.float32)
        np_array = self._pool.device_to_host(cl_buf, dtype=np_dtype, shape=shape)
        return torch.from_numpy(np_array.copy())

    def allocate_output(
        self,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
    ) -> CLBuffer:
        """Allocate an empty output buffer for a given tensor shape/dtype."""
        np_dtype = _TORCH_TO_NP.get(dtype, np.float32)
        nbytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
        return self._pool.allocate(nbytes, np_dtype, shape)

    def free_buffer(self, cl_buf: CLBuffer) -> None:
        """Return a buffer to the pool."""
        self._pool.free(cl_buf)

    # ── Kernel launching ─────────────────────────────────────────

    def _compute_global_size(self, n: int, local_size: int = 256) -> int:
        """Round up n to the nearest multiple of local_size."""
        return ((n + local_size - 1) // local_size) * local_size

    def run_elementwise_binary(
        self,
        kernel_name: str,
        a_buf: CLBuffer,
        b_buf: CLBuffer,
        out_buf: CLBuffer,
        n: int,
    ) -> None:
        """Run a binary element-wise kernel: out = op(a, b)."""
        queue = get_queue()
        kernel = self._registry.get_kernel("elementwise.cl", kernel_name)
        global_size = (self._compute_global_size(n),)
        local_size = (min(256, n),) if n >= 256 else None
        kernel(queue, global_size, local_size,
               a_buf.buffer, b_buf.buffer, out_buf.buffer, np.int32(n))

    def run_elementwise_unary(
        self,
        kernel_name: str,
        a_buf: CLBuffer,
        out_buf: CLBuffer,
        n: int,
    ) -> None:
        """Run a unary element-wise kernel: out = op(a)."""
        queue = get_queue()
        kernel = self._registry.get_kernel("elementwise.cl", kernel_name)
        global_size = (self._compute_global_size(n),)
        local_size = (min(256, n),) if n >= 256 else None
        kernel(queue, global_size, local_size,
               a_buf.buffer, out_buf.buffer, np.int32(n))

    def run_elementwise_scalar(
        self,
        kernel_name: str,
        a_buf: CLBuffer,
        scalar: float,
        out_buf: CLBuffer,
        n: int,
    ) -> None:
        """Run a scalar element-wise kernel: out = op(a, scalar)."""
        queue = get_queue()
        kernel = self._registry.get_kernel("elementwise.cl", kernel_name)
        global_size = (self._compute_global_size(n),)
        local_size = (min(256, n),) if n >= 256 else None
        kernel(queue, global_size, local_size,
               a_buf.buffer, np.float32(scalar), out_buf.buffer, np.int32(n))

    def run_activation(
        self,
        kernel_name: str,
        a_buf: CLBuffer,
        out_buf: CLBuffer,
        n: int,
        **extra_args,
    ) -> None:
        """Run an activation kernel."""
        queue = get_queue()
        kernel = self._registry.get_kernel("activation.cl", kernel_name)
        global_size = (self._compute_global_size(n),)
        local_size = (min(256, n),) if n >= 256 else None

        args = [a_buf.buffer]
        for v in extra_args.values():
            args.append(np.float32(v))
        args.extend([out_buf.buffer, np.int32(n)])
        kernel(queue, global_size, local_size, *args)

    def run_matmul(
        self,
        a_buf: CLBuffer,
        b_buf: CLBuffer,
        out_buf: CLBuffer,
        M: int,
        N: int,
        K: int,
        use_tiled: bool = True,
    ) -> None:
        """Run matrix multiplication: C[M,N] = A[M,K] @ B[K,N]."""
        queue = get_queue()
        device_info = get_device_info()

        if use_tiled and M >= 16 and N >= 16 and K >= 16:
            tile_size = 16
            if device_info["local_mem_size_kb"] < 8:
                tile_size = 8  # Smaller tiles for limited local memory

            kernel = self._registry.get_kernel(
                "matmul.cl", "matmul_tiled_f32",
                build_options=f"-DTILE_SIZE={tile_size}"
            )
            global_size = (
                self._compute_global_size(M, tile_size),
                self._compute_global_size(N, tile_size),
            )
            local_size = (tile_size, tile_size)
        else:
            kernel = self._registry.get_kernel("matmul.cl", "matmul_naive_f32")
            global_size = (
                self._compute_global_size(M, 16),
                self._compute_global_size(N, 16),
            )
            local_size = None

        kernel(queue, global_size, local_size,
               a_buf.buffer, b_buf.buffer, out_buf.buffer,
               np.int32(M), np.int32(N), np.int32(K))

    def run_reduction(
        self,
        kernel_name: str,
        a_buf: CLBuffer,
        out_buf: CLBuffer,
        n: int,
    ) -> None:
        """Run a reduction kernel (sum, max, min).

        Uses a two-pass approach: first reduce within workgroups,
        then reduce the workgroup results on CPU (simple and correct).
        """
        queue = get_queue()
        kernel = self._registry.get_kernel("reduction.cl", kernel_name)

        local_size = min(256, n)
        num_groups = (n + local_size - 1) // local_size
        global_size = num_groups * local_size

        # Allocate partial results buffer
        partial_buf = self._pool.allocate(num_groups * 4)  # float32 = 4 bytes

        kernel(queue, (global_size,), (local_size,),
               a_buf.buffer, partial_buf.buffer,
               cl.LocalMemory(local_size * 4),
               np.int32(n))

        # Read partial results and finish on CPU
        partials = self._pool.device_to_host(partial_buf, np.float32, (num_groups,))
        self._pool.free(partial_buf)

        if kernel_name == "sum_f32":
            result = np.array([partials.sum()], dtype=np.float32)
        elif kernel_name == "max_f32":
            result = np.array([partials.max()], dtype=np.float32)
        elif kernel_name == "min_f32":
            result = np.array([partials.min()], dtype=np.float32)
        else:
            result = np.array([partials.sum()], dtype=np.float32)

        self._pool.host_to_device(result, out_buf)

    def run_softmax(
        self,
        a_buf: CLBuffer,
        out_buf: CLBuffer,
        rows: int,
        cols: int,
    ) -> None:
        """Run row-wise softmax."""
        queue = get_queue()
        kernel = self._registry.get_kernel("reduction.cl", "softmax_f32")
        global_size = (self._compute_global_size(rows),)
        local_size = None
        kernel(queue, global_size, local_size,
               a_buf.buffer, out_buf.buffer,
               np.int32(rows), np.int32(cols))

    def run_fill(self, out_buf: CLBuffer, value: float, n: int) -> None:
        """Fill a buffer with a constant value."""
        queue = get_queue()
        kernel = self._registry.get_kernel("elementwise.cl", "fill_f32")
        global_size = (self._compute_global_size(n),)
        local_size = (min(256, n),) if n >= 256 else None
        kernel(queue, global_size, local_size,
               out_buf.buffer, np.float32(value), np.int32(n))

    def run_transpose(
        self,
        a_buf: CLBuffer,
        out_buf: CLBuffer,
        M: int,
        N: int,
    ) -> None:
        """Transpose a matrix: out[N,M] = a[M,N]^T."""
        queue = get_queue()
        kernel = self._registry.get_kernel("matmul.cl", "transpose_f32")
        global_size = (
            self._compute_global_size(M, 16),
            self._compute_global_size(N, 16),
        )
        local_size = None
        kernel(queue, global_size, local_size,
               a_buf.buffer, out_buf.buffer,
               np.int32(M), np.int32(N))


# ── Module-level singleton ───────────────────────────────────────────
_global_engine: OpenCLEngine | None = None


def get_engine() -> OpenCLEngine:
    """Return the global compute engine."""
    global _global_engine
    if _global_engine is None:
        _global_engine = OpenCLEngine()
    return _global_engine
