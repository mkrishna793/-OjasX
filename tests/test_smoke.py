"""
TorchCL Smoke Test — Verifies all core operations work correctly
by comparing OpenCL results against CPU PyTorch results.
"""

import sys
import time
import torch
import numpy as np

# Must import before torchcl to suppress output during test header
print("=" * 60)
print("  TorchCL V1 - Comprehensive Smoke Test")
print("=" * 60)
print()

import torchcl

passed = 0
failed = 0
errors = []


def check(name: str, cl_result, cpu_expected, atol=1e-4):
    """Compare OpenCL result to CPU expected value."""
    global passed, failed
    try:
        if isinstance(cl_result, torch.Tensor) and torchcl.is_opencl_tensor(cl_result):
            cl_result = torchcl.to_cpu(cl_result)

        if isinstance(cpu_expected, torch.Tensor):
            ok = torch.allclose(cl_result.float(), cpu_expected.float(), atol=atol)
        else:
            ok = abs(float(cl_result) - float(cpu_expected)) < atol

        if ok:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            errors.append(name)
            print(f"  [FAIL] {name}")
            print(f"         Got:      {cl_result.flatten()[:5]}")
            print(f"         Expected: {cpu_expected.flatten()[:5]}")
    except Exception as e:
        failed += 1
        errors.append(f"{name}: {e}")
        print(f"  [ERROR] {name}: {e}")


# ── Test 1: Data Movement ────────────────────────────────────────────
print("\n--- Data Movement ---")

a_cpu = torch.randn(100, 100)
a_cl = torchcl.to_opencl(a_cpu)
a_back = torchcl.to_cpu(a_cl)
check("CPU -> OpenCL -> CPU round-trip", a_back, a_cpu)

check("is_opencl_tensor (True)", torchcl.is_opencl_tensor(a_cl), True)
check("is_opencl_tensor (False)", torchcl.is_opencl_tensor(a_cpu), False)

# ── Test 2: Tensor Creation ──────────────────────────────────────────
print("\n--- Tensor Creation ---")

z = torchcl.zeros(3, 3)
check("zeros(3,3)", torchcl.to_cpu(z), torch.zeros(3, 3))

o = torchcl.ones(4, 4)
check("ones(4,4)", torchcl.to_cpu(o), torch.ones(4, 4))

f = torchcl.full(2, 3, fill_value=7.0)
check("full(2,3, val=7)", torchcl.to_cpu(f), torch.full((2, 3), 7.0))

r = torchcl.randn(5, 5)
r_cpu = torchcl.to_cpu(r)
check("randn(5,5) shape", torch.tensor(list(r_cpu.shape)), torch.tensor([5, 5]))

# ── Test 3: Arithmetic ───────────────────────────────────────────────
print("\n--- Arithmetic ---")

a = torch.randn(256, 256)
b = torch.randn(256, 256)
a_cl = torchcl.to_opencl(a)
b_cl = torchcl.to_opencl(b)

check("add", torchcl.add(a_cl, b_cl), a + b)
check("sub", torchcl.sub(a_cl, b_cl), a - b)
check("mul", torchcl.mul(a_cl, b_cl), a * b)
check("div", torchcl.div(a_cl, b_cl), a / b, atol=1e-3)
check("neg", torchcl.neg(a_cl), -a)

a_pos = torch.rand(100, 100) + 0.1  # positive values for log/sqrt
a_pos_cl = torchcl.to_opencl(a_pos)
check("abs", torchcl.abs_(a_cl), torch.abs(a))
check("exp", torchcl.exp(a_cl), torch.exp(a), atol=1e-3)
check("log", torchcl.log(a_pos_cl), torch.log(a_pos), atol=1e-3)
check("sqrt", torchcl.sqrt(a_pos_cl), torch.sqrt(a_pos), atol=1e-3)

# ── Test 4: Activations ──────────────────────────────────────────────
print("\n--- Activations ---")

x = torch.randn(128, 128)
x_cl = torchcl.to_opencl(x)

check("relu", torchcl.relu(x_cl), torch.relu(x))
check("sigmoid", torchcl.sigmoid(x_cl), torch.sigmoid(x), atol=1e-3)
check("tanh", torchcl.tanh_(x_cl), torch.tanh(x), atol=1e-3)
check("gelu", torchcl.gelu(x_cl), torch.nn.functional.gelu(x), atol=1e-2)
check("silu", torchcl.silu(x_cl), torch.nn.functional.silu(x), atol=1e-3)
check("leaky_relu", torchcl.leaky_relu(x_cl, 0.01),
      torch.nn.functional.leaky_relu(x, 0.01), atol=1e-3)

# Softmax
s = torch.randn(8, 16)
s_cl = torchcl.to_opencl(s)
check("softmax", torchcl.softmax(s_cl), torch.softmax(s, dim=-1), atol=1e-3)

# ── Test 5: Matrix Operations ────────────────────────────────────────
print("\n--- Matrix Operations ---")

m1 = torch.randn(64, 128)
m2 = torch.randn(128, 32)
m1_cl = torchcl.to_opencl(m1)
m2_cl = torchcl.to_opencl(m2)
check("matmul(64x128 @ 128x32)", torchcl.matmul(m1_cl, m2_cl), m1 @ m2, atol=1e-2)

m3 = torch.randn(16, 32)
m3_cl = torchcl.to_opencl(m3)
check("transpose(16x32)", torchcl.transpose(m3_cl), m3.T)

# ── Test 6: Reductions ───────────────────────────────────────────────
print("\n--- Reductions ---")

r = torch.randn(1024)
r_cl = torchcl.to_opencl(r)
check("sum", torchcl.to_cpu(torchcl.sum_(r_cl)).item(), r.sum().item(), atol=0.5)
check("mean", torchcl.to_cpu(torchcl.mean(r_cl)).item(), r.mean().item(), atol=0.1)
check("max", torchcl.to_cpu(torchcl.max_(r_cl)).item(), r.max().item(), atol=1e-3)
check("min", torchcl.to_cpu(torchcl.min_(r_cl)).item(), r.min().item(), atol=1e-3)

# ── Test 7: Larger matmul (performance sanity check) ─────────────────
print("\n--- Performance (matmul 512x512) ---")

big_a = torch.randn(512, 512)
big_b = torch.randn(512, 512)
big_a_cl = torchcl.to_opencl(big_a)
big_b_cl = torchcl.to_opencl(big_b)

start = time.time()
result_cl = torchcl.matmul(big_a_cl, big_b_cl)
torchcl.synchronize()
cl_time = time.time() - start

start = time.time()
result_cpu = big_a @ big_b
cpu_time = time.time() - start

check("matmul 512x512 correctness", result_cl, result_cpu, atol=0.5)
print(f"  [INFO] OpenCL time: {cl_time*1000:.1f} ms")
print(f"  [INFO] CPU time:    {cpu_time*1000:.1f} ms")

# ── Summary ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print(f"  RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

if errors:
    print("\n  Failed tests:")
    for e in errors:
        print(f"    - {e}")

if failed == 0:
    print("\n  ALL TESTS PASSED! TorchCL V1 is working on your GPU!")
    print(f"  Device: {torchcl.get_device_info()['name']}")

sys.exit(0 if failed == 0 else 1)
