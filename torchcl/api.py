"""
TorchCL Public API — High-level functions for OpenCL tensor operations.

This is what users interact with. All functions accept PyTorch tensors,
run the computation on OpenCL, and return PyTorch tensors.

Usage:
    import torchcl
    x = torchcl.to_opencl(torch.randn(100, 100))
    y = torchcl.to_opencl(torch.randn(100, 100))
    z = torchcl.add(x, y)
    result = torchcl.to_cpu(z)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from torchcl.ops.engine import get_engine
from torchcl.runtime.memory import CLBuffer, get_buffer_pool
from torchcl.runtime.context import synchronize as _sync

# ── Internal storage: maps tensor data_ptr → CLBuffer ────────────────
# Since we can't actually allocate on a real custom device without C++,
# we use a shadow-tensor approach: the "real" data lives in OpenCL buffers,
# and we keep a CPU-side tensor as a handle/placeholder.
_opencl_buffers: dict[int, CLBuffer] = {}
_tensor_id_counter = 0


def _next_id() -> int:
    global _tensor_id_counter
    _tensor_id_counter += 1
    return _tensor_id_counter


def _make_handle(shape: tuple, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, int]:
    """Create a CPU placeholder tensor and assign it a unique ID."""
    handle = torch.empty(1, dtype=dtype)  # tiny placeholder
    tid = _next_id()
    handle._torchcl_id = tid  # type: ignore[attr-defined]
    handle._torchcl_shape = shape  # type: ignore[attr-defined]
    handle._torchcl_dtype = dtype  # type: ignore[attr-defined]
    return handle, tid


def _get_buf(tensor: torch.Tensor) -> CLBuffer:
    """Get the OpenCL buffer for a TorchCL tensor handle."""
    tid = getattr(tensor, "_torchcl_id", None)
    if tid is None or tid not in _opencl_buffers:
        raise ValueError(
            "This tensor is not on the OpenCL device. "
            "Use torchcl.to_opencl(tensor) first."
        )
    return _opencl_buffers[tid]


def _get_shape(tensor: torch.Tensor) -> tuple:
    return getattr(tensor, "_torchcl_shape", tensor.shape)


def _get_dtype(tensor: torch.Tensor) -> torch.dtype:
    return getattr(tensor, "_torchcl_dtype", tensor.dtype)


def _wrap_output(cl_buf: CLBuffer, shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Wrap an OpenCL buffer as a TorchCL tensor handle."""
    handle, tid = _make_handle(shape, dtype)
    _opencl_buffers[tid] = cl_buf
    return handle


def is_opencl_tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is stored on OpenCL."""
    tid = getattr(tensor, "_torchcl_id", None)
    return tid is not None and tid in _opencl_buffers


# ── Data movement ────────────────────────────────────────────────────

def to_opencl(tensor: torch.Tensor) -> torch.Tensor:
    """Move a CPU tensor to OpenCL device."""
    if is_opencl_tensor(tensor):
        return tensor

    engine = get_engine()
    cl_buf = engine.tensor_to_buffer(tensor)
    return _wrap_output(cl_buf, tuple(tensor.shape), tensor.dtype)


def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Move an OpenCL tensor back to CPU."""
    if not is_opencl_tensor(tensor):
        return tensor

    engine = get_engine()
    shape = _get_shape(tensor)
    dtype = _get_dtype(tensor)
    return engine.buffer_to_tensor(_get_buf(tensor), shape, dtype)


def synchronize() -> None:
    """Wait for all OpenCL operations to complete."""
    _sync()


# ── Tensor creation ──────────────────────────────────────────────────

def zeros(*shape, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a zero-filled tensor on OpenCL."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    n = int(np.prod(shape))
    engine = get_engine()
    cl_buf = engine.allocate_output(shape, dtype)
    engine.run_fill(cl_buf, 0.0, n)
    return _wrap_output(cl_buf, shape, dtype)


def ones(*shape, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a ones-filled tensor on OpenCL."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    n = int(np.prod(shape))
    engine = get_engine()
    cl_buf = engine.allocate_output(shape, dtype)
    engine.run_fill(cl_buf, 1.0, n)
    return _wrap_output(cl_buf, shape, dtype)


def full(*shape, fill_value: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a constant-filled tensor on OpenCL."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    n = int(np.prod(shape))
    engine = get_engine()
    cl_buf = engine.allocate_output(shape, dtype)
    engine.run_fill(cl_buf, fill_value, n)
    return _wrap_output(cl_buf, shape, dtype)


def randn(*shape, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a random normal tensor on OpenCL."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    cpu_tensor = torch.randn(*shape, dtype=dtype)
    return to_opencl(cpu_tensor)


# ── Arithmetic operations ────────────────────────────────────────────

def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise addition on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(shape)
    engine.run_elementwise_binary("add_f32", _get_buf(a), _get_buf(b), out_buf, n)
    return _wrap_output(out_buf, shape)


def sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise subtraction on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(shape)
    engine.run_elementwise_binary("sub_f32", _get_buf(a), _get_buf(b), out_buf, n)
    return _wrap_output(out_buf, shape)


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise multiplication on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(shape)
    engine.run_elementwise_binary("mul_f32", _get_buf(a), _get_buf(b), out_buf, n)
    return _wrap_output(out_buf, shape)


def div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise division on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(shape)
    engine.run_elementwise_binary("div_f32", _get_buf(a), _get_buf(b), out_buf, n)
    return _wrap_output(out_buf, shape)


def neg(a: torch.Tensor) -> torch.Tensor:
    """Element-wise negation on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(shape)
    engine.run_elementwise_unary("neg_f32", _get_buf(a), out_buf, n)
    return _wrap_output(out_buf, shape)


def abs_(a: torch.Tensor) -> torch.Tensor:
    """Element-wise absolute value on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(shape)
    engine.run_elementwise_unary("abs_f32", _get_buf(a), out_buf, n)
    return _wrap_output(out_buf, shape)


def exp(a: torch.Tensor) -> torch.Tensor:
    """Element-wise exp on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(shape)
    engine.run_elementwise_unary("exp_f32", _get_buf(a), out_buf, n)
    return _wrap_output(out_buf, shape)


def log(a: torch.Tensor) -> torch.Tensor:
    """Element-wise log on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(shape)
    engine.run_elementwise_unary("log_f32", _get_buf(a), out_buf, n)
    return _wrap_output(out_buf, shape)


def sqrt(a: torch.Tensor) -> torch.Tensor:
    """Element-wise sqrt on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(shape)
    engine.run_elementwise_unary("sqrt_f32", _get_buf(a), out_buf, n)
    return _wrap_output(out_buf, shape)


# ── Activation functions ─────────────────────────────────────────────

def relu(a: torch.Tensor) -> torch.Tensor:
    """ReLU activation on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(shape)
    engine.run_activation("relu_f32", _get_buf(a), out_buf, n)
    return _wrap_output(out_buf, shape)


def sigmoid(a: torch.Tensor) -> torch.Tensor:
    """Sigmoid activation on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(shape)
    engine.run_activation("sigmoid_f32", _get_buf(a), out_buf, n)
    return _wrap_output(out_buf, shape)


def tanh_(a: torch.Tensor) -> torch.Tensor:
    """Tanh activation on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(shape)
    engine.run_activation("tanh_f32", _get_buf(a), out_buf, n)
    return _wrap_output(out_buf, shape)


def gelu(a: torch.Tensor) -> torch.Tensor:
    """GELU activation on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(shape)
    engine.run_activation("gelu_f32", _get_buf(a), out_buf, n)
    return _wrap_output(out_buf, shape)


def silu(a: torch.Tensor) -> torch.Tensor:
    """SiLU activation on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(shape)
    engine.run_activation("silu_f32", _get_buf(a), out_buf, n)
    return _wrap_output(out_buf, shape)


def leaky_relu(a: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """LeakyReLU activation on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output(shape)
    engine.run_activation("leaky_relu_f32", _get_buf(a), out_buf, n, neg_slope=negative_slope)
    return _wrap_output(out_buf, shape)


def softmax(a: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax on OpenCL (along last dimension)."""
    engine = get_engine()
    shape = _get_shape(a)
    if len(shape) == 1:
        rows, cols = 1, shape[0]
    else:
        rows = int(np.prod(shape[:-1]))
        cols = shape[-1]
    out_buf = engine.allocate_output(shape)
    engine.run_softmax(_get_buf(a), out_buf, rows, cols)
    return _wrap_output(out_buf, shape)


# ── Matrix operations ────────────────────────────────────────────────

def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication on OpenCL: C = A @ B."""
    engine = get_engine()
    a_shape = _get_shape(a)
    b_shape = _get_shape(b)

    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError(f"matmul requires 2D tensors, got {a_shape} and {b_shape}")

    M, K = a_shape
    K2, N = b_shape
    if K != K2:
        raise ValueError(f"matmul dimension mismatch: {a_shape} @ {b_shape}")

    out_shape = (M, N)
    out_buf = engine.allocate_output(out_shape)
    engine.run_matmul(_get_buf(a), _get_buf(b), out_buf, M, N, K)
    return _wrap_output(out_buf, out_shape)


def transpose(a: torch.Tensor) -> torch.Tensor:
    """Transpose a 2D tensor on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    if len(shape) != 2:
        raise ValueError(f"transpose requires 2D tensor, got {shape}")
    M, N = shape
    out_shape = (N, M)
    out_buf = engine.allocate_output(out_shape)
    engine.run_transpose(_get_buf(a), out_buf, M, N)
    return _wrap_output(out_buf, out_shape)


# ── Reduction operations ─────────────────────────────────────────────

def sum_(a: torch.Tensor) -> torch.Tensor:
    """Sum all elements on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output((1,))
    engine.run_reduction("sum_f32", _get_buf(a), out_buf, n)
    return _wrap_output(out_buf, (1,))


def mean(a: torch.Tensor) -> torch.Tensor:
    """Mean of all elements on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    # Sum then divide
    sum_buf = engine.allocate_output((1,))
    engine.run_reduction("sum_f32", _get_buf(a), sum_buf, n)
    out_buf = engine.allocate_output((1,))
    engine.run_elementwise_scalar("mul_scalar_f32", sum_buf, 1.0 / n, out_buf, 1)
    engine.free_buffer(sum_buf)
    return _wrap_output(out_buf, (1,))


def max_(a: torch.Tensor) -> torch.Tensor:
    """Max of all elements on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output((1,))
    engine.run_reduction("max_f32", _get_buf(a), out_buf, n)
    return _wrap_output(out_buf, (1,))


def min_(a: torch.Tensor) -> torch.Tensor:
    """Min of all elements on OpenCL."""
    engine = get_engine()
    shape = _get_shape(a)
    n = int(np.prod(shape))
    out_buf = engine.allocate_output((1,))
    engine.run_reduction("min_f32", _get_buf(a), out_buf, n)
    return _wrap_output(out_buf, (1,))
