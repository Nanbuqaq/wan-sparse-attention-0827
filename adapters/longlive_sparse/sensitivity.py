"""Layer/head history-sensitivity proxies from captured real Q/K/V tensors."""

from __future__ import annotations

import math

import torch


def history_head_sensitivity(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_chunk_size: int = 128,
) -> list[dict[str, float | int]]:
    """Compute bounded-memory history-only Attention diagnostics per head."""

    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("Q/K/V must be [B,T,H,D] and K/V must share shape")
    if query.shape[0] != 1 or key.shape[0] != 1:
        raise ValueError("sensitivity probe currently requires batch size one")
    if query.shape[2:] != key.shape[2:]:
        raise ValueError("Q/K/V head and dimension axes must match")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    heads, dim = query.shape[2], query.shape[3]
    records = []
    for head in range(heads):
        q = query[0, :, head].float()
        k = key[0, :, head].float()
        v = value[0, :, head].float()
        output_square = 0.0
        entropy_sum = 0.0
        max_probability_sum = 0.0
        queries = 0
        for start in range(0, q.shape[0], query_chunk_size):
            part = q[start : start + query_chunk_size]
            probability = torch.softmax(part @ k.T / math.sqrt(dim), dim=-1)
            output = probability @ v
            output_square += float(output.square().sum())
            entropy_sum += float(
                (-(probability.clamp_min(1e-12) * probability.clamp_min(1e-12).log()).sum(dim=-1)).sum()
            )
            max_probability_sum += float(probability.amax(dim=-1).sum())
            queries += part.shape[0]
        records.append(
            {
                "head": head,
                "query_tokens": queries,
                "history_tokens": int(k.shape[0]),
                "history_output_rms": math.sqrt(output_square / max(1, queries * dim)),
                "attention_entropy_mean": entropy_sum / max(1, queries),
                "attention_max_probability_mean": max_probability_sum / max(1, queries),
            }
        )
    return records
