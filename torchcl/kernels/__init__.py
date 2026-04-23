"""Kernel package — loads and compiles OpenCL kernel sources."""

from .registry import KernelRegistry, get_kernel_registry

__all__ = ["KernelRegistry", "get_kernel_registry"]
