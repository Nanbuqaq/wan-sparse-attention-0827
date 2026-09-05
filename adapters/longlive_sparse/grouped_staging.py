"""Batched staging experiment for the *same* grouped FA2 attention graph.

This does not eliminate per-group KV replication or implement KVOut. It removes
the Python/CUDA per-group materialization loop and its intermediate KV copies.
Only coordinate metadata is reusable; no Q/K/V or output is cached here.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
import torch

from .route_plan import HistoryRoutePlan
from .profiling import profiled


@dataclass(frozen=True)
class GroupedPackingRecipe:
    route_sha: str
    geometry: tuple[int, int, int, int, int]  # B,Q,H,exactT,unionT
    query_indices: torch.Tensor
    key_indices: torch.Tensor
    cu_query: torch.Tensor
    cu_key: torch.Tensor
    query_lengths: tuple[int, ...]
    key_lengths: tuple[int, ...]

    @property
    def metadata_bytes(self):
        return sum(value.numel()*value.element_size() for value in
                   (self.query_indices, self.key_indices, self.cu_query, self.cu_key))


@profiled('attention/build_batched_staging_recipe')
def build_grouped_packing_recipe(plan: HistoryRoutePlan, *, exact_tokens: int,
                                 union_tokens: int) -> GroupedPackingRecipe:
    if exact_tokens != plan.exact_k_tokens or union_tokens != plan.union_frame_ids.shape[-1]:
        raise ValueError('recipe geometry must match exact and compact union lengths')
    labels = plan.query_labels.detach().cpu().numpy()
    membership = plan.group_union_indices.detach().cpu().numpy()
    counts = plan.group_history_counts.detach().cpu().numpy()
    valid = (plan.union_frame_ids.detach().cpu().numpy() >= 0).sum(-1)
    batch, heads, queries = labels.shape
    query_parts, key_parts, q_lens, k_lens = [], [], [], []
    total_k = exact_tokens + union_tokens
    for b in range(batch):
        for h in range(heads):
            exact = (b*total_k + np.arange(exact_tokens, dtype=np.int64))*heads+h
            for group in range(int(labels[b,h].max())+1):
                query_tokens = np.flatnonzero(labels[b,h] == group)
                if query_tokens.size == 0:
                    continue
                count = int(counts[b,h,group])
                indices = membership[b,h,group,:count].astype(np.int64, copy=False)
                if count and (indices.min() < 0 or indices.max() >= valid[b,h]):
                    raise IndexError('group references padded or invalid union tokens')
                history = (b*total_k+exact_tokens+indices)*heads+h
                query_parts.append((b*queries+query_tokens)*heads+h)
                key_parts.append(np.concatenate((exact, history)))
                q_lens.append(int(query_tokens.size))
                k_lens.append(exact_tokens+count)
    if not q_lens or min(k_lens) < 1:
        raise ValueError('every active query group needs at least one key')
    q = np.concatenate(query_parts)
    if q.size != batch*heads*queries or np.unique(q).size != q.size:
        raise ValueError('query recipe must be an exact permutation')
    if sum(k_lens) >= 2**31 or sum(q_lens) >= 2**31:
        raise ValueError('FA2 sequence offsets exceed int32')
    return GroupedPackingRecipe(plan.digest(), (batch, queries, heads, exact_tokens, union_tokens),
        torch.from_numpy(q), torch.from_numpy(np.concatenate(key_parts)),
        torch.from_numpy(np.cumsum([0]+q_lens, dtype=np.int32)),
        torch.from_numpy(np.cumsum([0]+k_lens, dtype=np.int32)), tuple(q_lens), tuple(k_lens))


@profiled('attention/batched_qkv_staging')
def pack_grouped_qkv(recipe, query, exact_key, exact_value, history_key, history_value):
    batch, queries, heads, exact, union = recipe.geometry
    dim = query.shape[-1]
    if query.shape != (batch, queries, heads, dim):
        raise ValueError('query geometry mismatch')
    for key, value, tokens in ((exact_key, exact_value, exact), (history_key, history_value, union)):
        if key.shape != (batch, tokens, heads, dim) or value.shape != key.shape:
            raise ValueError('KV geometry mismatch')
        if key.dtype != query.dtype or value.dtype != query.dtype or key.device != query.device or value.device != query.device:
            raise ValueError('all attention tensors must share dtype and device')
    q_indices = recipe.query_indices.to(query.device)
    k_indices = recipe.key_indices.to(query.device)
    q = query.reshape(-1, dim).index_select(0, q_indices).unsqueeze(1)
    # The shared tensors are created once; final varlen storage is gathered once.
    # Peak staging no longer holds hundreds of intermediate group K/V tensors.
    k = torch.cat((exact_key, history_key), dim=1).reshape(-1, dim).index_select(0, k_indices).unsqueeze(1)
    v = torch.cat((exact_value, history_value), dim=1).reshape(-1, dim).index_select(0, k_indices).unsqueeze(1)
    return q, k, v, q_indices


@profiled('attention/batched_grouped_fa2_complete')
def execute_batched_grouped_fa2(query, exact_key, exact_value, history_key, history_value,
                               plan, *, recipe=None):
    """Replay-only pilot. Entire recipe/pack/kernel/restore time is included."""
    if not query.is_cuda:
        raise RuntimeError('real CUDA and FA2 are required; no dense fallback')
    import flash_attn
    from .backends import BackendResult
    start = time.perf_counter()
    if recipe is None:
        recipe = build_grouped_packing_recipe(plan, exact_tokens=exact_key.shape[1],
                                              union_tokens=history_key.shape[1])
    elif recipe.route_sha != plan.digest():
        raise ValueError('stale route recipe')
    q, k, v, q_indices = pack_grouped_qkv(recipe, query, exact_key, exact_value, history_key, history_value)
    output = flash_attn.flash_attn_varlen_func(q=q, k=k, v=v,
        cu_seqlens_q=recipe.cu_query.to(query.device), cu_seqlens_k=recipe.cu_key.to(query.device),
        max_seqlen_q=max(recipe.query_lengths), max_seqlen_k=max(recipe.key_lengths),
        dropout_p=0., causal=False)
    restored = torch.empty(query.numel()//query.shape[-1], query.shape[-1], device=query.device, dtype=query.dtype)
    restored.index_copy_(0, q_indices, output[:,0])
    torch.cuda.synchronize(query.device)
    elapsed = (time.perf_counter()-start)*1000
    pairs = sum(qn*kn for qn,kn in zip(recipe.query_lengths, recipe.key_lengths))
    return BackendResult(restored.reshape(query.shape), 'batched_grouped_fa2_replay', elapsed,
                         pairs, pairs, 0, plan.digest())
