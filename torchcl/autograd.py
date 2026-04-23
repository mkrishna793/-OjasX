"""
OjasX V2 — Autograd Functions
Custom torch.autograd.Function classes that enable gradient-based
training entirely on OpenCL GPU.
"""

from __future__ import annotations
import torch
import torchcl
from torchcl.api import to_opencl, to_cpu, is_opencl_tensor


class OpenCLMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return torchcl.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = torchcl.matmul(grad_output, torchcl.transpose(b))
        grad_b = torchcl.matmul(torchcl.transpose(a), grad_output)
        return grad_a, grad_b


class OpenCLReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return torchcl.relu(a)

    @staticmethod
    def backward(ctx, grad_output):
        a, = ctx.saved_tensors
        a_cpu = to_cpu(a)
        grad_cpu = to_cpu(grad_output)
        return to_opencl(grad_cpu * (a_cpu > 0).float())


class OpenCLSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        out = torchcl.sigmoid(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        s = to_cpu(out)
        g = to_cpu(grad_output)
        return to_opencl(g * s * (1 - s))


class OpenCLSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        out = torchcl.softmax(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        s = to_cpu(out)
        g = to_cpu(grad_output)
        dot = (g * s).sum(dim=-1, keepdim=True)
        return to_opencl(s * (g - dot))


class OpenCLLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        w_t = torchcl.transpose(weight)
        out = torchcl.matmul(input, w_t)
        if bias is not None:
            out_cpu = to_cpu(out) + to_cpu(bias)
            out = to_opencl(out_cpu)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = torchcl.matmul(grad_output, weight)
        grad_weight = torchcl.matmul(torchcl.transpose(grad_output), input)
        grad_bias = None
        if bias is not None:
            grad_bias = to_opencl(to_cpu(grad_output).sum(dim=0))
        return grad_input, grad_weight, grad_bias


# Convenience wrappers
def ocl_matmul(a, b): return OpenCLMatmul.apply(a, b)
def ocl_relu(a): return OpenCLReLU.apply(a)
def ocl_sigmoid(a): return OpenCLSigmoid.apply(a)
def ocl_softmax(a): return OpenCLSoftmax.apply(a)
def ocl_linear(x, w, b=None): return OpenCLLinear.apply(x, w, b)
