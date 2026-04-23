"""
OjasX V2 — nn.Module wrappers for OpenCL layers.
Drop-in replacements: Linear, ReLU, Conv2d, MaxPool2d, BatchNorm1d,
Dropout, Flatten, Sequential, Softmax.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import numpy as np

import torchcl
from torchcl.autograd import ocl_linear, ocl_relu, ocl_sigmoid, ocl_softmax
from torchcl.api import to_opencl, to_cpu, is_opencl_tensor
from torchcl.ops.engine import get_engine
from torchcl.runtime.memory import get_buffer_pool
from torchcl.runtime.context import get_queue
from torchcl.kernels.registry import get_kernel_registry


class Linear(nn.Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        w = torch.randn(out_f, in_f) * math.sqrt(2.0 / in_f)
        self.weight = to_opencl(w)
        self.bias = to_opencl(torch.zeros(out_f)) if bias else None

    def forward(self, x):
        return ocl_linear(x, self.weight, self.bias)

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def __repr__(self):
        return f"ocl.Linear({self.in_f}, {self.out_f})"


class ReLU(nn.Module):
    def forward(self, x): return ocl_relu(x)
    def parameters(self): return []


class Sigmoid(nn.Module):
    def forward(self, x): return ocl_sigmoid(x)
    def parameters(self): return []


class Softmax(nn.Module):
    def forward(self, x): return ocl_softmax(x)
    def parameters(self): return []


class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        kH, kW = self.ks
        w = torch.randn(out_ch, in_ch * kH * kW) * math.sqrt(2.0 / (in_ch * kH * kW))
        self.weight = to_opencl(w)
        self.bias = to_opencl(torch.zeros(out_ch)) if bias else None

    def forward(self, x):
        x_cpu = to_cpu(x)
        B, C, H, W = x_cpu.shape
        kH, kW = self.ks
        sH, sW = self.stride
        pH, pW = self.padding
        outH = (H + 2 * pH - kH) // sH + 1
        outW = (W + 2 * pW - kW) // sW + 1

        engine = get_engine()
        pool = get_buffer_pool()
        queue = get_queue()
        reg = get_kernel_registry()
        im2col_k = reg.get_kernel("conv.cl", "im2col_f32")
        col_size = C * kH * kW * outH * outW

        outputs = []
        for b in range(B):
            img_buf = engine.tensor_to_buffer(x_cpu[b].contiguous())
            col_buf = pool.allocate(col_size * 4)
            gs = (engine._compute_global_size(col_size),)
            im2col_k(queue, gs, None, img_buf.buffer, col_buf.buffer,
                      np.int32(C), np.int32(H), np.int32(W),
                      np.int32(kH), np.int32(kW), np.int32(pH), np.int32(pW),
                      np.int32(sH), np.int32(sW), np.int32(outH), np.int32(outW))

            w_buf = torchcl.api._get_buf(self.weight)
            out_buf = pool.allocate(self.out_ch * outH * outW * 4)
            engine.run_matmul(w_buf, col_buf, out_buf, self.out_ch, outH * outW, C * kH * kW)

            if self.bias is not None:
                bk = reg.get_kernel("conv.cl", "bias_add_f32")
                bb = torchcl.api._get_buf(self.bias)
                sp = outH * outW
                bk(queue, (engine._compute_global_size(self.out_ch * sp),), None,
                   out_buf.buffer, bb.buffer, np.int32(self.out_ch), np.int32(sp))

            r = pool.device_to_host(out_buf, np.float32, (self.out_ch, outH, outW))
            outputs.append(torch.from_numpy(r.copy()))
            pool.free(col_buf); pool.free(out_buf); pool.free(img_buf)

        return to_opencl(torch.stack(outputs))

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def __repr__(self):
        return f"ocl.Conv2d({self.in_ch}, {self.out_ch}, ks={self.ks})"


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride or self.ks
        if not isinstance(self.stride, tuple): self.stride = (self.stride, self.stride)

    def forward(self, x):
        x_cpu = to_cpu(x)
        B, C, H, W = x_cpu.shape
        kH, kW = self.ks
        sH, sW = self.stride
        outH, outW = (H - kH) // sH + 1, (W - kW) // sW + 1
        engine = get_engine()
        pool = get_buffer_pool()
        queue = get_queue()
        mpk = get_kernel_registry().get_kernel("conv.cl", "maxpool2d_f32")
        outputs = []
        for b in range(B):
            ib = engine.tensor_to_buffer(x_cpu[b].contiguous())
            sz = C * outH * outW
            ob = pool.allocate(sz * 4); xb = pool.allocate(sz * 4)
            mpk(queue, (engine._compute_global_size(sz),), None,
                ib.buffer, ob.buffer, xb.buffer,
                np.int32(C), np.int32(H), np.int32(W),
                np.int32(kH), np.int32(kW), np.int32(sH), np.int32(sW),
                np.int32(outH), np.int32(outW))
            r = pool.device_to_host(ob, np.float32, (C, outH, outW))
            outputs.append(torch.from_numpy(r.copy()))
            pool.free(ib); pool.free(ob); pool.free(xb)
        return to_opencl(torch.stack(outputs))

    def parameters(self): return []


class BatchNorm1d(nn.Module):
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = to_opencl(torch.ones(features))
        self.beta = to_opencl(torch.zeros(features))

    def forward(self, x):
        xc = to_cpu(x); g = to_cpu(self.gamma); b = to_cpu(self.beta)
        m = xc.mean(0); v = xc.var(0, unbiased=False)
        return to_opencl(g * (xc - m) / torch.sqrt(v + self.eps) + b)

    def parameters(self): return [self.gamma, self.beta]


class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p; self._train = True

    def forward(self, x):
        if not self._train: return x
        xc = to_cpu(x)
        return to_opencl(xc * (torch.rand_like(xc) > self.p).float() / (1 - self.p))

    def train(self, m=True): self._train = m; return self
    def eval(self): return self.train(False)
    def parameters(self): return []


class Flatten(nn.Module):
    def forward(self, x):
        xc = to_cpu(x); return to_opencl(xc.view(xc.size(0), -1))
    def parameters(self): return []


class Sequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for l in self.layers: x = l(x)
        return x

    def parameters(self):
        p = []
        for l in self.layers: p.extend(l.parameters())
        return p

    def __repr__(self):
        return "ocl.Sequential(\n" + "\n".join(f"  ({i}): {l}" for i,l in enumerate(self.layers)) + "\n)"
