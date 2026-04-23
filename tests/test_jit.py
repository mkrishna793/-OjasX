"""
TorchCL JIT Compiler Test — Tests fused kernel generation and execution.
"""

import torch
import numpy as np
import pyopencl as cl

import torchcl
from torchcl.jit.compiler import get_jit_compiler
from torchcl.jit.tuner import get_auto_tuner
from torchcl.runtime.memory import get_buffer_pool
from torchcl.runtime.context import get_queue

print()
print("=" * 60)
print("  TorchCL JIT Compiler Test")
print("=" * 60)

passed = 0
failed = 0


def check(name, result, expected, atol=1e-3):
    global passed, failed
    if np.allclose(result, expected, atol=atol):
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")
        print(f"         Got:      {result[:5]}")
        print(f"         Expected: {expected[:5]}")


# ── Test 1: Fused unary chain ────────────────────────────────────────
print("\n--- Fused Unary Chains ---")

jit = get_jit_compiler()
pool = get_buffer_pool()
queue = get_queue()

# Test: relu(sigmoid(x))
n = 1024
x = np.random.randn(n).astype(np.float32)
x_buf = pool.host_to_device(x)
out_buf = pool.allocate(n * 4)

jit.fuse_elementwise_chain(["sigmoid", "relu"], n, [x_buf.buffer], out_buf.buffer)
result = pool.device_to_host(out_buf, np.float32, (n,))
expected = np.maximum(0, 1.0 / (1.0 + np.exp(-x)))
check("fused sigmoid -> relu", result, expected)

# Test: exp(neg(x))
jit.fuse_elementwise_chain(["neg", "exp"], n, [x_buf.buffer], out_buf.buffer)
result = pool.device_to_host(out_buf, np.float32, (n,))
expected = np.exp(-x)
check("fused neg -> exp", result, expected)

# Test: abs(tanh(x))
jit.fuse_elementwise_chain(["tanh", "abs"], n, [x_buf.buffer], out_buf.buffer)
result = pool.device_to_host(out_buf, np.float32, (n,))
expected = np.abs(np.tanh(x))
check("fused tanh -> abs", result, expected)

# ── Test 2: Fused binary + unary ─────────────────────────────────────
print("\n--- Fused Binary + Unary Chains ---")

a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
a_buf = pool.host_to_device(a)
b_buf = pool.host_to_device(b)

# Test: relu(a + b) — most common fusion in neural networks!
jit.fuse_binary_then_unary("add", ["relu"], n, a_buf.buffer, b_buf.buffer, out_buf.buffer)
result = pool.device_to_host(out_buf, np.float32, (n,))
expected = np.maximum(0, a + b)
check("fused add -> relu", result, expected)

# Test: sigmoid(a * b)
jit.fuse_binary_then_unary("mul", ["sigmoid"], n, a_buf.buffer, b_buf.buffer, out_buf.buffer)
result = pool.device_to_host(out_buf, np.float32, (n,))
expected = 1.0 / (1.0 + np.exp(-(a * b)))
check("fused mul -> sigmoid", result, expected)

# ── Test 3: Auto-tuner ──────────────────────────────────────────────
print("\n--- Auto-Tuner ---")

tuner = get_auto_tuner()
print(f"\n{tuner.summary()}")

wg = tuner.optimal_workgroup_1d(100000)
print(f"\n  Workgroup for 100K elements: {wg}")
assert wg > 0 and (wg & (wg - 1)) == 0, "Workgroup must be power of 2"
passed += 1
print(f"  [PASS] workgroup is power of 2: {wg}")

tile = tuner.optimal_tile_size(512, 512, 512)
print(f"  Tile size for 512x512 matmul: {tile}")
assert tile in (4, 8, 16), f"Unexpected tile size: {tile}"
passed += 1
print(f"  [PASS] tile size is valid: {tile}")

# ── Test 4: Kernel cache ────────────────────────────────────────────
print("\n--- Kernel Cache ---")

from torchcl.jit.cache import get_kernel_cache
cache = get_kernel_cache()

# Run the same fusion again to trigger a cache hit
jit.fuse_elementwise_chain(["sigmoid", "relu"], n, [x_buf.buffer], out_buf.buffer)

stats = cache.stats()
print(f"  Cache stats: {stats}")
assert stats["hits"] > 0, "Expected cache hits after repeated JIT call"
passed += 1
print(f"  [PASS] cache has hits: {stats['hits']}")

# ── Summary ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"  JIT RESULTS: {passed} passed, {failed} failed")
print("=" * 60)
