"""Operators package — dispatched PyTorch operations running on OpenCL."""

from .engine import OpenCLEngine, get_engine

__all__ = ["OpenCLEngine", "get_engine"]
