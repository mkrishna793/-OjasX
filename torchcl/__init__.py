"""
TorchCL — Universal OpenCL Backend for PyTorch
================================================

Enables PyTorch to run tensor operations on ANY OpenCL-capable GPU:
AMD, Intel, Qualcomm, ARM Mali, and more.

Usage:
    import torchcl

    # Check device
    print(torchcl.get_device_info())

    # Move tensors to OpenCL
    x = torchcl.to_opencl(torch.randn(512, 512))
    y = torchcl.to_opencl(torch.randn(512, 512))

    # Compute on GPU
    z = torchcl.matmul(x, y)
    z = torchcl.relu(z)

    # Get results back
    result = torchcl.to_cpu(z)
"""

__version__ = "0.1.0"
__author__ = "TorchCL Contributors"

# Initialize OpenCL runtime on import
from torchcl.runtime.context import (
    init_opencl,
    get_device_info,
    is_available,
    synchronize,
)

# Public API — tensor operations
from torchcl.api import (
    # Data movement
    to_opencl,
    to_cpu,
    is_opencl_tensor,
    # Creation
    zeros,
    ones,
    full,
    randn,
    # Arithmetic
    add,
    sub,
    mul,
    div,
    neg,
    abs_,
    exp,
    log,
    sqrt,
    # Activations
    relu,
    sigmoid,
    tanh_,
    gelu,
    silu,
    leaky_relu,
    softmax,
    # Matrix
    matmul,
    transpose,
    # Reductions
    sum_,
    mean,
    max_,
    min_,
)

# Auto-initialize on import
try:
    init_opencl()
    _info = get_device_info()
    print(f"[TorchCL] OK - Initialized on: {_info['name']}")
    print(f"[TorchCL]   OpenCL {_info['version']} | "
          f"{_info['max_compute_units']} CUs | "
          f"{_info['global_mem_size_mb']} MB")
except Exception as e:
    print(f"[TorchCL] FAILED - Could not initialize OpenCL: {e}")
    print("[TorchCL]   Falling back to CPU-only mode.")

# V3 Native Integration
try:
    from torchcl.tensor import apply_monkeypatches
    apply_monkeypatches()
except Exception as e:
    print(f"[TorchCL] V3 Native Integration Failed: {e}")

__all__ = [
    # Info
    "get_device_info",
    "is_available",
    "synchronize",
    # Data movement
    "to_opencl",
    "to_cpu",
    "is_opencl_tensor",
    # Creation
    "zeros",
    "ones",
    "full",
    "randn",
    # Arithmetic
    "add", "sub", "mul", "div", "neg", "abs_", "exp", "log", "sqrt",
    # Activations
    "relu", "sigmoid", "tanh_", "gelu", "silu", "leaky_relu", "softmax",
    # Matrix
    "matmul", "transpose",
    # Reductions
    "sum_", "mean", "max_", "min_",
]
