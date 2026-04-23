"""
TorchCL Basic Usage Example
============================
Demonstrates how to use TorchCL for GPU-accelerated tensor operations
on ANY OpenCL-capable device (AMD, Intel, Qualcomm, ARM, etc.)
"""

import torch
import torchcl

print()
print("=" * 60)
print("  TorchCL - Running PyTorch on OpenCL!")
print("=" * 60)

# ── Device Info ──────────────────────────────────────────────────────
info = torchcl.get_device_info()
print(f"\n  GPU: {info['name']}")
print(f"  OpenCL: {info['version']}")
print(f"  Memory: {info['global_mem_size_mb']} MB")
print(f"  Compute Units: {info['max_compute_units']}")

# ── Basic Arithmetic ─────────────────────────────────────────────────
print("\n--- Basic Arithmetic ---")

# Create tensors on CPU, move to OpenCL
a = torchcl.to_opencl(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
b = torchcl.to_opencl(torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0]))

# Math on GPU
c = torchcl.add(a, b)       # [11, 22, 33, 44, 55]
d = torchcl.mul(a, b)       # [10, 40, 90, 160, 250]
e = torchcl.relu(torchcl.sub(a, torchcl.to_opencl(torch.tensor([3.0]*5))))

print(f"  a + b = {torchcl.to_cpu(c).tolist()}")
print(f"  a * b = {torchcl.to_cpu(d).tolist()}")
print(f"  relu(a - 3) = {torchcl.to_cpu(e).tolist()}")

# ── Matrix Multiplication ────────────────────────────────────────────
print("\n--- Matrix Multiplication ---")

import time

sizes = [128, 256, 512]
for size in sizes:
    m1 = torchcl.to_opencl(torch.randn(size, size))
    m2 = torchcl.to_opencl(torch.randn(size, size))

    start = time.time()
    result = torchcl.matmul(m1, m2)
    torchcl.synchronize()
    elapsed = (time.time() - start) * 1000

    result_cpu = torchcl.to_cpu(result)
    print(f"  {size}x{size} matmul: {elapsed:.1f} ms (result shape: {list(result_cpu.shape)})")

# ── Neural Network Simulation ────────────────────────────────────────
print("\n--- Simple Neural Network Forward Pass ---")

# Simulate a 3-layer MLP on OpenCL
batch_size = 32
input_dim = 784   # MNIST-like
hidden_dim = 256
output_dim = 10

# Random weights (normally you'd train these)
W1 = torchcl.to_opencl(torch.randn(input_dim, hidden_dim) * 0.01)
W2 = torchcl.to_opencl(torch.randn(hidden_dim, hidden_dim) * 0.01)
W3 = torchcl.to_opencl(torch.randn(hidden_dim, output_dim) * 0.01)

# Random input batch
x = torchcl.to_opencl(torch.randn(batch_size, input_dim))

# Forward pass — entirely on GPU!
start = time.time()
h1 = torchcl.relu(torchcl.matmul(x, W1))       # Layer 1: Linear + ReLU
h2 = torchcl.relu(torchcl.matmul(h1, W2))      # Layer 2: Linear + ReLU
out = torchcl.softmax(torchcl.matmul(h2, W3))   # Layer 3: Linear + Softmax
torchcl.synchronize()
elapsed = (time.time() - start) * 1000

result = torchcl.to_cpu(out)
print(f"  Input:  {batch_size} x {input_dim}")
print(f"  Output: {list(result.shape)}")
print(f"  Time:   {elapsed:.1f} ms")
print(f"  Prediction (sample 0): class {result[0].argmax().item()} "
      f"(confidence: {result[0].max().item():.2%})")

# ── Summary ──────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  TorchCL V1 is running on your GPU!")
print(f"  No CUDA needed. Just OpenCL on {info['name']}.")
print("=" * 60)
