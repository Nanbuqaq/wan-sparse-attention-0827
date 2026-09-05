"""Isolated full-QKV teacher evaluation; never imported by online routing."""

from __future__ import annotations

import torch

from .route_plan import HistoryRoutePlan, map_union_coordinates


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_chunk_size: int,
) -> torch.Tensor:
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    scale = query.shape[-1] ** -0.5
    outputs = []
    key_fp32 = key.float()
    value_fp32 = value.float()
    for start in range(0, query.shape[0], query_chunk_size):
        query_chunk = query[start : start + query_chunk_size].float()
        probability = torch.softmax(query_chunk @ key_fp32.T * scale, dim=-1)
        outputs.append(probability @ value_fp32)
    return torch.cat(outputs, dim=0)


def dense_history_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_chunk_size: int = 64,
) -> torch.Tensor:
    """FP32 history-only teacher over full candidate K/V."""

    if query.ndim != 4 or key.ndim != 4 or key.shape != value.shape:
        raise ValueError("Q/K/V must be [B,T,H,D] and K/V must match")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("Q/K/V batch, head and dimension axes must match")
    output = torch.empty_like(query, dtype=torch.float32)
    for batch_index in range(query.shape[0]):
        for head_index in range(query.shape[2]):
            output[batch_index, :, head_index] = _attention(
                query[batch_index, :, head_index],
                key[batch_index, :, head_index],
                value[batch_index, :, head_index],
                query_chunk_size=query_chunk_size,
            )
    return output


def routed_history_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    frame_ids: torch.Tensor,
    token_ids: torch.Tensor,
    plan: HistoryRoutePlan,
    *,
    query_chunk_size: int = 64,
) -> torch.Tensor:
    """FP32 routed output using exactly the supplied HistoryRoutePlan."""

    if plan.query_tokens != query.shape[1]:
        raise ValueError("route query length does not match Q")
    union_to_dense = map_union_coordinates(plan, frame_ids, token_ids)
    output = torch.empty_like(query, dtype=torch.float32)
    for batch_index in range(query.shape[0]):
        for head_index in range(query.shape[2]):
            groups = int(plan.query_labels[batch_index, head_index].max()) + 1
            for group in range(groups):
                query_indices = torch.nonzero(
                    plan.query_labels[batch_index, head_index] == group,
                    as_tuple=False,
                ).flatten().to(query.device)
                count = int(plan.group_history_counts[batch_index, head_index, group])
                union_indices = plan.group_union_indices[
                    batch_index, head_index, group, :count
                ]
                dense_indices = union_to_dense[
                    batch_index, head_index
                ].index_select(0, union_indices).to(key.device)
                if dense_indices.numel() == 0:
                    raise ValueError("history-only routed evaluation requires non-empty groups")
                output[batch_index, query_indices, head_index] = _attention(
                    query[batch_index, query_indices, head_index],
                    key[batch_index, :, head_index].index_select(0, dense_indices),
                    value[batch_index, :, head_index].index_select(0, dense_indices),
                    query_chunk_size=query_chunk_size,
                )
    return output


def output_error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate outputs must share shape")
    reference = reference.float().reshape(-1)
    candidate = candidate.float().reshape(-1)
    delta = candidate - reference
    relative_l2 = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(
        reference
    ).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        reference.unsqueeze(0), candidate.unsqueeze(0)
    )[0]
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(relative_l2),
        "one_minus_cosine": float(1.0 - cosine),
    }
