"""TorchCL Runtime — OpenCL platform, device, context, and queue management."""

from .context import (
    init_opencl,
    get_context,
    get_queue,
    get_device,
    get_device_info,
    synchronize,
    is_available,
)
from .memory import CLBufferPool

__all__ = [
    "init_opencl",
    "get_context",
    "get_queue",
    "get_device",
    "get_device_info",
    "synchronize",
    "is_available",
    "CLBufferPool",
]
