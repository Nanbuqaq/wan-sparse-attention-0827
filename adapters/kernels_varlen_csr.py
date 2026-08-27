"""CSR-indexed clean-room variable-block sparse attention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _varlen_csr_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    row_ptr,
    col_idx,
    chunk_row,
    chunk_start,
    chunk_valid,
    k_cum,
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
    stride_kcb,
    stride_kch,
    stride_kcc,
    heads,
    q_clusters,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    program = tl.program_id(0)
    flat_row = tl.load(chunk_row + program)
    q_start = tl.load(chunk_start + program)
    q_valid = tl.load(chunk_valid + program)
    head_batch = flat_row // q_clusters
    q_cluster = flat_row - head_batch * q_clusters
    head = head_batch % heads
    batch = head_batch // heads

    rows = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    valid_rows = rows < q_valid
    q_offsets = (
        batch * stride_qb
        + head * stride_qh
        + (q_start + rows[:, None]) * stride_qs
        + dims[None, :] * stride_qd
    )
    q = tl.load(q_ptr + q_offsets, mask=valid_rows[:, None], other=0.0)
    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    edge = tl.load(row_ptr + flat_row)
    edge_end = tl.load(row_ptr + flat_row + 1)
    while edge < edge_end:
        k_cluster = tl.load(col_idx + edge)
        k_base = batch * stride_kcb + head * stride_kch
        k_start = tl.load(k_cum + k_base + k_cluster * stride_kcc)
        k_end = tl.load(k_cum + k_base + (k_cluster + 1) * stride_kcc)
        offset = 0
        while k_start + offset < k_end:
            cols = tl.arange(0, BLOCK_N)
            absolute = k_start + offset + cols
            valid_cols = absolute < k_end
            k_offsets = (
                batch * stride_kb
                + head * stride_kh
                + absolute[:, None] * stride_ks
                + dims[None, :] * stride_kd
            )
            v_offsets = (
                batch * stride_vb
                + head * stride_vh
                + absolute[:, None] * stride_vs
                + dims[None, :] * stride_vd
            )
            k = tl.load(k_ptr + k_offsets, mask=valid_cols[:, None], other=0.0)
            v = tl.load(v_ptr + v_offsets, mask=valid_cols[:, None], other=0.0)
            scores = tl.dot(q, tl.trans(k)) * scale
            scores = tl.where(valid_rows[:, None] & valid_cols[None, :], scores, -float("inf"))
            tile_max = tl.max(scores, axis=1)
            next_max = tl.maximum(running_max, tile_max)
            previous_scale = tl.exp(running_max - next_max)
            probabilities = tl.exp(scores - next_max[:, None])
            probabilities = tl.where(valid_rows[:, None] & valid_cols[None, :], probabilities, 0.0)
            running_sum = running_sum * previous_scale + tl.sum(probabilities, axis=1)
            accumulator = accumulator * previous_scale[:, None] + tl.dot(
                probabilities.to(v.dtype), v
            )
            running_max = next_max
            offset += BLOCK_N
        edge += 1

    denominator = tl.where(running_sum > 0, running_sum, 1.0)
    result = accumulator / denominator[:, None]
    result = tl.where(running_sum[:, None] > 0, result, 0.0)
    out_offsets = (
        batch * stride_ob
        + head * stride_oh
        + (q_start + rows[:, None]) * stride_os
        + dims[None, :] * stride_od
    )
    tl.store(out_ptr + out_offsets, result.to(tl.bfloat16), mask=valid_rows[:, None])


def _csr_layout(block_map: torch.Tensor, q_sizes: torch.Tensor, k_sizes: torch.Tensor, block_m: int):
    batch, heads, q_clusters, _ = block_map.shape
    flat_map = block_map.reshape(batch * heads, q_clusters, -1)
    row_counts = flat_map.sum(dim=-1, dtype=torch.int32).reshape(-1)
    if torch.any(row_counts == 0):
        raise ValueError("CSR varlen requires at least one active K cluster per Q cluster")
    row_ptr = torch.cat(
        (torch.zeros(1, dtype=torch.int32, device=block_map.device), row_counts.cumsum(0)),
        dim=0,
    )
    col_idx = flat_map.nonzero(as_tuple=False)[:, 2].to(torch.int32).contiguous()
    q_cum = torch.cat(
        (torch.zeros_like(q_sizes[..., :1]), q_sizes.cumsum(dim=-1)), dim=-1
    ).to(torch.int32)
    k_cum = torch.cat(
        (torch.zeros_like(k_sizes[..., :1]), k_sizes.cumsum(dim=-1)), dim=-1
    ).to(torch.int32)
    flat_q = q_sizes.reshape(-1).to(torch.int64)
    chunks = torch.div(flat_q + block_m - 1, block_m, rounding_mode="floor")
    row_ids = torch.repeat_interleave(
        torch.arange(flat_q.numel(), device=block_map.device, dtype=torch.int64), chunks
    )
    starts = chunks.cumsum(0) - chunks
    ordinal = torch.arange(row_ids.numel(), device=block_map.device) - torch.repeat_interleave(starts, chunks)
    bh = torch.div(row_ids, q_clusters, rounding_mode="floor")
    qc = row_ids % q_clusters
    batch_ids = torch.div(bh, heads, rounding_mode="floor")
    head_ids = bh % heads
    absolute_start = q_cum[batch_ids, head_ids, qc].to(torch.int64) + ordinal * block_m
    valid = torch.minimum(
        torch.full_like(ordinal, block_m), flat_q.index_select(0, row_ids) - ordinal * block_m
    )
    return (
        row_ptr.contiguous(),
        col_idx,
        row_ids.to(torch.int32).contiguous(),
        absolute_start.to(torch.int32).contiguous(),
        valid.to(torch.int32).contiguous(),
        k_cum.contiguous(),
    )


def prepare_varlen_csr(
    block_map: torch.Tensor,
    q_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    *,
    block_m: int = 64,
):
    return _csr_layout(block_map, q_sizes, k_sizes, block_m)


def varlen_csr_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_map: torch.Tensor,
    q_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    *,
    block_m: int = 64,
    block_n: int = 32,
    prepared=None,
) -> torch.Tensor:
    if query.shape != key.shape or query.shape != value.shape or query.ndim != 4:
        raise ValueError("CSR varlen expects matching [B,H,S,D]")
    if query.dtype is not torch.bfloat16 or not query.is_cuda:
        raise TypeError("CSR varlen requires CUDA BF16 inputs")
    if block_m not in (32, 64, 128) or block_n not in (16, 32, 64, 128):
        raise ValueError("unsupported CSR tile size")
    if block_map.dtype is not torch.bool:
        raise TypeError("CSR block map must be boolean")
    if torch.any(q_sizes < 0) or torch.any(k_sizes < 0):
        raise ValueError("cluster sizes must be non-negative")
    if not torch.all(q_sizes.sum(-1) == query.shape[2]):
        raise ValueError("Q cluster sizes do not sum to sequence length")
    if not torch.all(k_sizes.sum(-1) == key.shape[2]):
        raise ValueError("K cluster sizes do not sum to sequence length")

    row_ptr, col_idx, chunk_row, chunk_start, chunk_valid, k_cum = (
        prepared
        if prepared is not None
        else _csr_layout(block_map, q_sizes, k_sizes, block_m)
    )
    output = torch.empty_like(query)
    _varlen_csr_kernel[(chunk_row.numel(),)](
        query,
        key,
        value,
        output,
        row_ptr,
        col_idx,
        chunk_row,
        chunk_start,
        chunk_valid,
        k_cum,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        *k_cum.stride(),
        query.shape[1],
        q_sizes.shape[-1],
        query.shape[-1] ** -0.5,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=query.shape[-1],
        num_warps=4,
        num_stages=2,
    )
    return output
