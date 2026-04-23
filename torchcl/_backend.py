"""
torch.compile Backend for TorchCL — Captures the FX computation graph
and executes it on OpenCL with operator fusion.

Usage:
    import torchcl
    model = torch.compile(model, backend="opencl")
    output = model(input)
"""

from __future__ import annotations

import torch
import torch.fx
import numpy as np

from torchcl.jit.compiler import get_jit_compiler
from torchcl.ops.engine import get_engine
from torchcl.runtime.memory import get_buffer_pool


# ── Map FX op names to TorchCL operations ────────────────────────────
_FX_OP_MAP = {
    # Arithmetic
    torch.ops.aten.add.Tensor: "add",
    torch.ops.aten.sub.Tensor: "sub",
    torch.ops.aten.mul.Tensor: "mul",
    torch.ops.aten.div.Tensor: "div",
    torch.ops.aten.neg.default: "neg",
    torch.ops.aten.abs.default: "abs",
    torch.ops.aten.exp.default: "exp",
    torch.ops.aten.log.default: "log",
    torch.ops.aten.sqrt.default: "sqrt",
    # Activations
    torch.ops.aten.relu.default: "relu",
    torch.ops.aten.sigmoid.default: "sigmoid",
    torch.ops.aten.tanh.default: "tanh",
    torch.ops.aten.gelu.default: "gelu",
    torch.ops.aten.silu.default: "silu",
}

_FUSEABLE_UNARY = {"relu", "sigmoid", "tanh", "neg", "abs", "exp", "log", "sqrt", "gelu", "silu"}
_FUSEABLE_BINARY = {"add", "sub", "mul", "div"}


def _identify_fusion_chains(gm: torch.fx.GraphModule) -> list[list[torch.fx.Node]]:
    """Walk the FX graph and identify sequences of element-wise ops
    that can be fused into a single kernel."""
    chains = []
    visited = set()

    for node in gm.graph.nodes:
        if node in visited:
            continue
        if node.op != "call_function":
            continue

        op_name = _FX_OP_MAP.get(node.target)
        if op_name is None or op_name not in _FUSEABLE_UNARY:
            continue

        # Start a chain
        chain = [node]
        visited.add(node)
        current = node

        # Follow the chain: if the output feeds into another fuseable unary
        while len(current.users) == 1:
            user = list(current.users.keys())[0]
            user_op = _FX_OP_MAP.get(user.target) if user.op == "call_function" else None
            if user_op in _FUSEABLE_UNARY:
                chain.append(user)
                visited.add(user)
                current = user
            else:
                break

        if len(chain) >= 2:
            chains.append(chain)

    return chains


def opencl_backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    """The torch.compile backend entry point.

    Receives an FX GraphModule from TorchDynamo, analyzes it for fusion
    opportunities, and returns an optimized callable.

    For V1, we identify fusion chains and log them, but execute via
    the standard graph forward (the real fusion will happen in future
    versions with full kernel generation).
    """
    # Analyze the graph for fusion opportunities
    chains = _identify_fusion_chains(gm)
    if chains:
        fused_ops = [
            " -> ".join(_FX_OP_MAP.get(n.target, "?") for n in chain)
            for chain in chains
        ]
        for ops in fused_ops:
            print(f"[TorchCL JIT] Fusion opportunity: {ops}")

    # For V1: return the original forward function
    # Future: return a custom callable that uses fused OpenCL kernels
    return gm.forward


# ── Register with torch.compile ──────────────────────────────────────
try:
    from torch._dynamo import register_backend
    register_backend(name="opencl", compiler_fn=opencl_backend)
except ImportError:
    # Older PyTorch without Dynamo — skip registration
    pass
except Exception:
    pass
