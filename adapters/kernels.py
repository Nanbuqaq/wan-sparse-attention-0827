"""Strict fixed-length and variable-length sparse kernel execution."""

from __future__ import annotations

import time

import torch

from .routing import inverse_permute
from .types import RoutePlan
from .vendor import load_svoo_core
from .kernels_fixed64 import fixed64_sparse_attention
from .kernels_varlen_csr import prepare_varlen_csr, varlen_csr_attention


_FIXED_KERNEL = None
_VARLEN_KERNEL = None


def _timed_cuda(fn):
    if not torch.cuda.is_available():
        start = time.perf_counter()
        result = fn()
        return result, (time.perf_counter() - start) * 1000.0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result = fn()
    end_event.record()
    torch.cuda.synchronize()
    return result, float(start_event.elapsed_time(end_event))


def _fixed_kernel():
    global _FIXED_KERNEL
    if _FIXED_KERNEL is None:
        _FIXED_KERNEL = fixed64_sparse_attention
    return _FIXED_KERNEL


def _varlen_kernel():
    global _VARLEN_KERNEL
    if _VARLEN_KERNEL is None:
        _VARLEN_KERNEL = load_svoo_core().dynamic_block_sparse_fwd_triton
    return _VARLEN_KERNEL


def execute_route(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: RoutePlan,
) -> tuple[torch.Tensor, float, float]:
    if plan.backend == "fixed64_bf16":
        valid_k_sizes = plan.k_sizes[0, 0].to(torch.int32).contiguous()
        output, kernel_ms = _timed_cuda(
            lambda: _fixed_kernel()(query, key, value, plan.block_map, valid_k_sizes)
        )
        original_length = int(plan.metadata["original_length"])
        output = output[:, :, :original_length]
    elif plan.backend in {"varlen_triton", "varlen_triton_native"}:
        output, kernel_ms = _timed_cuda(
            lambda: _varlen_kernel()(
                query,
                key,
                value,
                plan.block_map,
                plan.q_sizes,
                plan.k_sizes,
            )
        )
    elif plan.backend == "varlen_triton_csr":
        backend_params = plan.metadata.get("backend_params", {})
        block_m = int(backend_params.get("block_m", 64))
        block_n = int(backend_params.get("block_n", 32))
        prepared, planner_ms = _timed_cuda(
            lambda: prepare_varlen_csr(
                plan.block_map,
                plan.q_sizes,
                plan.k_sizes,
                block_m=block_m,
            )
        )
        plan.planner_ms = planner_ms
        output, kernel_ms = _timed_cuda(
            lambda: varlen_csr_attention(
                query,
                key,
                value,
                plan.block_map,
                plan.q_sizes,
                plan.k_sizes,
                block_m=block_m,
                block_n=block_n,
                prepared=prepared,
            )
        )
    else:
        raise ValueError(f"unknown backend: {plan.backend}")

    inverse_ms = 0.0
    if plan.q_sorted_indices is not None:
        output, inverse_ms = _timed_cuda(
            lambda: inverse_permute(output, plan.q_sorted_indices)
        )
    if not torch.isfinite(output).all():
        raise FloatingPointError("sparse kernel produced NaN/Inf")
    return output, kernel_ms, inverse_ms
