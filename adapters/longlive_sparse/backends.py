"""Execution backends that replay an identical HistoryRoutePlan."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .route_plan import HistoryRoutePlan


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by environment preflight
    triton = None
    tl = None


@dataclass
class BackendResult:
    output: torch.Tensor
    backend: str
    elapsed_ms: float
    logical_pairs: int
    scheduled_pairs: int
    padding_pairs: int
    route_plan_sha256: str

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "elapsed_ms": self.elapsed_ms,
            "logical_pairs": self.logical_pairs,
            "scheduled_pairs": self.scheduled_pairs,
            "padding_pairs": self.padding_pairs,
            "route_plan_sha256": self.route_plan_sha256,
        }


def _sequences(
    query: torch.Tensor,
    exact_key: torch.Tensor,
    exact_value: torch.Tensor,
    history_key: torch.Tensor,
    history_value: torch.Tensor,
    plan: HistoryRoutePlan,
):
    batch, _, heads, _ = query.shape
    items = []
    for b in range(batch):
        for h in range(heads):
            labels = plan.query_labels[b, h].to(query.device)
            groups = int(labels.max()) + 1
            valid_union = int((plan.union_frame_ids[b, h] >= 0).sum())
            for group in range(groups):
                q_indices = torch.nonzero(labels == group, as_tuple=False).flatten()
                count = int(plan.group_history_counts[b, h, group])
                union_indices = plan.group_union_indices[b, h, group, :count].to(query.device)
                if count:
                    if int(union_indices.max()) >= valid_union:
                        raise IndexError("route plan references padded history union")
                    selected_k = history_key[b, :valid_union, h].index_select(0, union_indices)
                    selected_v = history_value[b, :valid_union, h].index_select(0, union_indices)
                    key = torch.cat((exact_key[b, :, h], selected_k), dim=0)
                    value = torch.cat((exact_value[b, :, h], selected_v), dim=0)
                else:
                    key = exact_key[b, :, h]
                    value = exact_value[b, :, h]
                items.append((b, h, q_indices, query[b, :, h].index_select(0, q_indices), key, value))
    return items


def _restore(items, outputs, shape, device, dtype):
    restored = torch.empty(shape, device=device, dtype=dtype)
    for (b, h, q_indices, *_), output in zip(items, outputs):
        restored[b, q_indices, h] = output
    return restored


def execute_grouped_fa2(
    query: torch.Tensor,
    exact_key: torch.Tensor,
    exact_value: torch.Tensor,
    history_key: torch.Tensor,
    history_value: torch.Tensor,
    plan: HistoryRoutePlan,
) -> BackendResult:
    items = _sequences(query, exact_key, exact_value, history_key, history_value, plan)
    start = time.perf_counter()
    outputs = []
    logical_pairs = 0
    if query.is_cuda:
        from wan.modules.attention import attention

        max_q = max(item[3].shape[0] for item in items)
        max_k = max(item[4].shape[0] for item in items)
        q_batch = torch.zeros((len(items), max_q, 1, query.shape[-1]), device=query.device, dtype=query.dtype)
        k_batch = torch.zeros((len(items), max_k, 1, query.shape[-1]), device=query.device, dtype=query.dtype)
        v_batch = torch.zeros_like(k_batch)
        q_lens, k_lens = [], []
        for index, item in enumerate(items):
            q, k, v = item[3], item[4], item[5]
            q_batch[index, : q.shape[0], 0] = q
            k_batch[index, : k.shape[0], 0] = k
            v_batch[index, : v.shape[0], 0] = v
            q_lens.append(q.shape[0])
            k_lens.append(k.shape[0])
            logical_pairs += q.shape[0] * k.shape[0]
        output_batch = attention(
            q_batch,
            k_batch,
            v_batch,
            q_lens=torch.tensor(q_lens, dtype=torch.int32, device=query.device),
            k_lens=torch.tensor(k_lens, dtype=torch.int32, device=query.device),
        )
        outputs = [output_batch[index, : q_lens[index], 0] for index in range(len(items))]
        torch.cuda.synchronize(query.device)
    else:
        for item in items:
            q, k, v = item[3], item[4], item[5]
            output = F.scaled_dot_product_attention(
                q.float().T.unsqueeze(0).transpose(1, 2),
                k.float().T.unsqueeze(0).transpose(1, 2),
                v.float().T.unsqueeze(0).transpose(1, 2),
            ).squeeze(0).squeeze(0)
            outputs.append(output.to(query.dtype))
            logical_pairs += q.shape[0] * k.shape[0]
    elapsed_ms = (time.perf_counter() - start) * 1000
    restored = _restore(items, outputs, query.shape, query.device, query.dtype)
    return BackendResult(
        output=restored,
        backend="grouped_fa2" if query.is_cuda else "grouped_sdpa_reference",
        elapsed_ms=elapsed_ms,
        logical_pairs=logical_pairs,
        scheduled_pairs=logical_pairs,
        padding_pairs=0,
        route_plan_sha256=plan.digest(),
    )


if triton is not None:
    @triton.jit
    def _fixed64_rect_kernel(
        Q, K, V, O, Q_LENS, K_LENS,
        stride_qb: tl.constexpr, stride_qs: tl.constexpr, stride_qd: tl.constexpr,
        stride_kb: tl.constexpr, stride_ks: tl.constexpr, stride_kd: tl.constexpr,
        stride_vb: tl.constexpr, stride_vs: tl.constexpr, stride_vd: tl.constexpr,
        stride_ob: tl.constexpr, stride_os: tl.constexpr, stride_od: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        MAX_K: tl.constexpr,
    ):
        sequence = tl.program_id(1)
        q_block = tl.program_id(0)
        q_len = tl.load(Q_LENS + sequence)
        k_len = tl.load(K_LENS + sequence)
        offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        q = tl.load(
            Q + sequence * stride_qb + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
            mask=(offs_m[:, None] < q_len) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
        scale = 1.0 / tl.sqrt(float(HEAD_DIM))
        for start_n in range(0, MAX_K, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(
                K + sequence * stride_kb + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                mask=(offs_n[:, None] < k_len) & (offs_d[None, :] < HEAD_DIM),
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k)) * scale
            scores = tl.where(offs_n[None, :] < k_len, scores, -float("inf"))
            row_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, row_max)
            alpha = tl.exp(m_i - m_new)
            probability = tl.exp(scores - m_new[:, None])
            probability = tl.where(offs_n[None, :] < k_len, probability, 0.0)
            v = tl.load(
                V + sequence * stride_vb + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                mask=(offs_n[:, None] < k_len) & (offs_d[None, :] < HEAD_DIM),
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(probability.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(probability, axis=1)
            m_i = m_new
        output = acc / l_i[:, None]
        tl.store(
            O + sequence * stride_ob + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
            output,
            mask=(offs_m[:, None] < q_len) & (offs_d[None, :] < HEAD_DIM),
        )


def execute_fixed64_rect(
    query: torch.Tensor,
    exact_key: torch.Tensor,
    exact_value: torch.Tensor,
    history_key: torch.Tensor,
    history_value: torch.Tensor,
    plan: HistoryRoutePlan,
) -> BackendResult:
    if not query.is_cuda or triton is None:
        raise RuntimeError("fixed64_rect requires CUDA and Triton")
    items = _sequences(query, exact_key, exact_value, history_key, history_value, plan)
    q_lens = [item[3].shape[0] for item in items]
    k_lens = [item[4].shape[0] for item in items]
    max_q = math.ceil(max(q_lens) / 64) * 64
    max_k = math.ceil(max(k_lens) / 64) * 64
    q_batch = torch.zeros((len(items), max_q, query.shape[-1]), device=query.device, dtype=query.dtype)
    k_batch = torch.zeros((len(items), max_k, query.shape[-1]), device=query.device, dtype=query.dtype)
    v_batch = torch.zeros_like(k_batch)
    for index, item in enumerate(items):
        q, k, v = item[3], item[4], item[5]
        q_batch[index, : q.shape[0]] = q
        k_batch[index, : k.shape[0]] = k
        v_batch[index, : v.shape[0]] = v
    output_batch = torch.empty_like(q_batch)
    q_lens_tensor = torch.tensor(q_lens, dtype=torch.int32, device=query.device)
    k_lens_tensor = torch.tensor(k_lens, dtype=torch.int32, device=query.device)
    start = time.perf_counter()
    grid = (max_q // 64, len(items))
    _fixed64_rect_kernel[grid](
        q_batch, k_batch, v_batch, output_batch, q_lens_tensor, k_lens_tensor,
        q_batch.stride(0), q_batch.stride(1), q_batch.stride(2),
        k_batch.stride(0), k_batch.stride(1), k_batch.stride(2),
        v_batch.stride(0), v_batch.stride(1), v_batch.stride(2),
        output_batch.stride(0), output_batch.stride(1), output_batch.stride(2),
        HEAD_DIM=query.shape[-1], BLOCK_M=64, BLOCK_N=64, MAX_K=max_k,
    )
    torch.cuda.synchronize(query.device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    outputs = [output_batch[index, : q_lens[index]] for index in range(len(items))]
    restored = _restore(items, outputs, query.shape, query.device, query.dtype)
    logical = sum(q * k for q, k in zip(q_lens, k_lens))
    scheduled = sum(math.ceil(q / 64) * 64 * math.ceil(k / 64) * 64 for q, k in zip(q_lens, k_lens))
    return BackendResult(
        output=restored,
        backend="fixed64_rect",
        elapsed_ms=elapsed_ms,
        logical_pairs=logical,
        scheduled_pairs=scheduled,
        padding_pairs=scheduled - logical,
        route_plan_sha256=plan.digest(),
    )


def execute_plan(
    backend: str,
    query: torch.Tensor,
    exact_key: torch.Tensor,
    exact_value: torch.Tensor,
    history_key: torch.Tensor,
    history_value: torch.Tensor,
    plan: HistoryRoutePlan,
) -> BackendResult:
    if backend in {"packed_fa2", "grouped_fa2"}:
        return execute_grouped_fa2(query, exact_key, exact_value, history_key, history_value, plan)
    if backend == "fixed64_rect":
        return execute_fixed64_rect(query, exact_key, exact_value, history_key, history_value, plan)
    if backend == "varlen_triton":
        raise NotImplementedError("rectangular varlen Triton gate has not passed yet")
    raise ValueError(f"unknown backend: {backend}")

