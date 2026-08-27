"""Clean-room BF16 fixed-64 block-sparse attention kernel.

This implementation was written for this workstream from the public online
softmax equations.  It does not contain source from the unlicensed local
``fp8-sparse-attn`` checkout.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fixed64_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    map_ptr,
    k_size_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    stride_mb,
    stride_mh,
    stride_mq,
    stride_mk,
    heads,
    scale,
    Q_BLOCKS: tl.constexpr,
    K_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    program = tl.program_id(0)
    q_block = program % Q_BLOCKS
    head_batch = program // Q_BLOCKS
    head = head_batch % heads
    batch = head_batch // heads

    rows = tl.arange(0, 64)
    dims = tl.arange(0, HEAD_DIM)
    q_start = q_block * 64
    q_offsets = (
        batch * stride_qb
        + head * stride_qh
        + (q_start + rows[:, None]) * stride_qs
        + dims[None, :] * stride_qd
    )
    q = tl.load(q_ptr + q_offsets)

    running_max = tl.full((64,), -float("inf"), tl.float32)
    running_sum = tl.zeros((64,), tl.float32)
    accumulator = tl.zeros((64, HEAD_DIM), tl.float32)
    map_base = batch * stride_mb + head * stride_mh + q_block * stride_mq

    for k_block in range(K_BLOCKS):
        active = tl.load(map_ptr + map_base + k_block * stride_mk)
        if active:
            valid_k = tl.load(k_size_ptr + k_block)
            cols = tl.arange(0, 64)
            k_start = k_block * 64
            k_offsets = (
                batch * stride_kb
                + head * stride_kh
                + (k_start + cols[:, None]) * stride_ks
                + dims[None, :] * stride_kd
            )
            v_offsets = (
                batch * stride_vb
                + head * stride_vh
                + (k_start + cols[:, None]) * stride_vs
                + dims[None, :] * stride_vd
            )
            valid_cols = cols < valid_k
            k = tl.load(k_ptr + k_offsets, mask=valid_cols[:, None], other=0.0)
            v = tl.load(v_ptr + v_offsets, mask=valid_cols[:, None], other=0.0)
            scores = tl.dot(q, tl.trans(k)) * scale
            scores = tl.where(valid_cols[None, :], scores, -float("inf"))

            tile_max = tl.max(scores, axis=1)
            next_max = tl.maximum(running_max, tile_max)
            previous_scale = tl.exp(running_max - next_max)
            probabilities = tl.exp(scores - next_max[:, None])
            probabilities = tl.where(valid_cols[None, :], probabilities, 0.0)
            running_sum = running_sum * previous_scale + tl.sum(probabilities, axis=1)
            accumulator = accumulator * previous_scale[:, None] + tl.dot(
                probabilities.to(v.dtype), v
            )
            running_max = next_max

    denominator = tl.where(running_sum > 0, running_sum, 1.0)
    output = accumulator / denominator[:, None]
    output = tl.where(running_sum[:, None] > 0, output, 0.0)
    out_offsets = (
        batch * stride_ob
        + head * stride_oh
        + (q_start + rows[:, None]) * stride_os
        + dims[None, :] * stride_od
    )
    tl.store(out_ptr + out_offsets, output.to(tl.bfloat16))


def fixed64_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_map: torch.Tensor,
    valid_k_sizes: torch.Tensor,
) -> torch.Tensor:
    if query.shape != key.shape or query.shape != value.shape or query.ndim != 4:
        raise ValueError("fixed64 expects matching [B,H,S,D] tensors")
    if query.dtype is not torch.bfloat16:
        raise TypeError("fixed64 clean-room kernel requires BF16")
    if not query.is_cuda or not block_map.is_cuda or not valid_k_sizes.is_cuda:
        raise ValueError("fixed64 clean-room kernel requires CUDA inputs")
    batch, heads, length, head_dim = query.shape
    if head_dim not in (32, 64, 128):
        raise ValueError(f"unsupported head dimension: {head_dim}")
    if length % 64:
        raise ValueError("fixed64 inputs must be padded to a multiple of 64")
    q_blocks = length // 64
    if block_map.shape[:3] != (batch, heads, q_blocks):
        raise ValueError("fixed64 block map shape mismatch")
    k_blocks = block_map.shape[-1]
    if k_blocks != q_blocks or valid_k_sizes.shape != (k_blocks,):
        raise ValueError("fixed64 currently expects square self-attention blocks")
    if torch.any(block_map.sum(dim=-1) == 0):
        raise ValueError("every fixed64 query row must select at least one key block")

    output = torch.empty_like(query)
    grid = (batch * heads * q_blocks,)
    _fixed64_kernel[grid](
        query,
        key,
        value,
        output,
        block_map,
        valid_k_sizes,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        *block_map.stride(),
        heads,
        head_dim**-0.5,
        Q_BLOCKS=q_blocks,
        K_BLOCKS=k_blocks,
        HEAD_DIM=head_dim,
        num_warps=4,
        num_stages=2,
    )
    return output
