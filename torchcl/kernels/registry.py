"""
Kernel Registry — Loads .cl source files, compiles them into OpenCL programs,
and provides fast access to individual kernels by name.

Compiled programs are cached so each .cl file is compiled only once.
"""

from __future__ import annotations

import pathlib
from typing import Any

import pyopencl as cl

from torchcl.runtime.context import get_context, get_device

# Directory containing .cl kernel source files
_KERNEL_DIR = pathlib.Path(__file__).parent


class KernelRegistry:
    """Loads, compiles, and caches OpenCL kernels."""

    def __init__(self) -> None:
        self._programs: dict[str, cl.Program] = {}
        self._kernels: dict[str, cl.Kernel] = {}

    def _compile_source(self, name: str, source: str, build_options: str = "") -> cl.Program:
        """Compile an OpenCL source string and cache the program."""
        if name in self._programs:
            return self._programs[name]

        ctx = get_context()
        program = cl.Program(ctx, source).build(options=build_options)
        self._programs[name] = program
        return program

    def _load_file(self, filename: str, build_options: str = "") -> cl.Program:
        """Load a .cl file from the kernels directory and compile it."""
        if filename in self._programs:
            return self._programs[filename]

        filepath = _KERNEL_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Kernel file not found: {filepath}")

        source = filepath.read_text(encoding="utf-8")
        return self._compile_source(filename, source, build_options)

    def get_kernel(self, filename: str, kernel_name: str, build_options: str = "") -> cl.Kernel:
        """Get a specific kernel from a .cl file.

        Args:
            filename:    Name of the .cl file (e.g. 'elementwise.cl')
            kernel_name: Name of the __kernel function (e.g. 'add_f32')
            build_options: Optional compiler flags (e.g. '-DTILE_SIZE=16')

        Returns:
            Compiled cl.Kernel ready to be enqueued.
        """
        cache_key = f"{filename}:{kernel_name}:{build_options}"
        if cache_key in self._kernels:
            return self._kernels[cache_key]

        program = self._load_file(filename, build_options)
        kernel = cl.Kernel(program, kernel_name)
        self._kernels[cache_key] = kernel
        return kernel

    def compile_source(self, name: str, source: str, kernel_name: str, build_options: str = "") -> cl.Kernel:
        """Compile a raw source string and return a kernel.

        Used by the JIT compiler to compile dynamically generated kernels.
        """
        cache_key = f"jit:{name}:{kernel_name}:{build_options}"
        if cache_key in self._kernels:
            return self._kernels[cache_key]

        program = self._compile_source(f"jit:{name}", source, build_options)
        kernel = cl.Kernel(program, kernel_name)
        self._kernels[cache_key] = kernel
        return kernel

    def clear_cache(self) -> None:
        """Clear all cached programs and kernels."""
        self._programs.clear()
        self._kernels.clear()


# ── Module-level singleton ───────────────────────────────────────────
_global_registry: KernelRegistry | None = None


def get_kernel_registry() -> KernelRegistry:
    """Return the global kernel registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = KernelRegistry()
    return _global_registry
