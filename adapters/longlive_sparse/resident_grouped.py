"""Exact grouped FA2 with resident metadata and direct separate-source packing.

This reduces control transfers and temporary packing; it does NOT eliminate
the final per-group KV replication or implement KV-major consumption.
"""
from dataclasses import replace
import time

import torch

from .grouped_staging import build_grouped_packing_recipe


class ResidentGroupedExecutor:
    def __init__(self):
        self.clear()
        self.hits = self.misses = 0

    def clear(self):
        self.identity = self.recipe = None

    @property
    def GPU_metadata_bytes(self):
        return self.recipe.metadata_bytes if self.recipe is not None else 0

    def prepare(self, plan, query, exact_key, history_key):
        route_sha = plan.digest()
        geometry = (query.shape[0], query.shape[1], query.shape[2], exact_key.shape[1], history_key.shape[1])
        identity = (route_sha, geometry, query.shape[-1], query.dtype, query.device)
        if identity == self.identity:
            self.hits += 1
            return self.recipe, 0
        recipe = build_grouped_packing_recipe(plan, exact_tokens=exact_key.shape[1], union_tokens=history_key.shape[1])
        self.recipe = replace(recipe, **{name: getattr(recipe, name).to(query.device) for name in
            ('query_indices', 'key_indices', 'cu_query', 'cu_key')})
        self.identity = identity
        self.misses += 1
        return self.recipe, recipe.metadata_bytes

    def pack(self, recipe, query, exact_key, exact_value, history_key, history_value):
        import triton
        from .grouped_pack_kernels import gather_queries, gather_separate_kv
        batch, queries, heads, exact, union = recipe.geometry
        dim = query.shape[-1]
        if query.shape != (batch, queries, heads, dim):
            raise ValueError('query geometry differs from recipe')
        for key, value, count in ((exact_key, exact_value, exact), (history_key, history_value, union)):
            if key.shape != (batch, count, heads, dim) or value.shape != key.shape:
                raise ValueError('KV geometry differs from recipe')
            if any(t.dtype != query.dtype or t.device != query.device for t in (key, value)):
                raise ValueError('Q/K/V dtype or device mismatch')
        q_count, k_count = sum(recipe.query_lengths), sum(recipe.key_lengths)
        q = torch.empty((q_count, 1, dim), dtype=query.dtype, device=query.device)
        k = torch.empty((k_count, 1, dim), dtype=query.dtype, device=query.device)
        v = torch.empty_like(k)
        rows, cols = 16, triton.next_power_of_2(dim)
        gather_queries[(triton.cdiv(q_count, rows),)](query, recipe.query_indices, q, q_count,
            queries, heads, dim, *query.stride(), rows, cols)
        gather_separate_kv[(triton.cdiv(k_count, rows),)](exact_key, exact_value, history_key, history_value,
            recipe.key_indices, k, v, k_count, exact, union, heads, dim,
            *exact_key.stride(), *exact_value.stride(), *history_key.stride(), *history_value.stride(), rows, cols)
        return q, k, v

    def execute(self, query, exact_key, exact_value, history_key, history_value, plan):
        if not query.is_cuda:
            raise RuntimeError('real CUDA required; no dense fallback')
        from flash_attn import flash_attn_varlen_func
        from .backends import BackendResult
        started = time.perf_counter()
        recipe, metadata_H2D_bytes = self.prepare(plan, query, exact_key, history_key)
        q, k, v = self.pack(recipe, query, exact_key, exact_value, history_key, history_value)
        output = flash_attn_varlen_func(q, k, v, recipe.cu_query, recipe.cu_key,
            max(recipe.query_lengths), max(recipe.key_lengths), dropout_p=0., causal=False)
        restored = torch.empty((query.numel()//query.shape[-1], query.shape[-1]), dtype=query.dtype, device=query.device)
        restored.index_copy_(0, recipe.query_indices, output[:, 0])
        torch.cuda.synchronize(query.device)
        pairs = sum(qn*kn for qn, kn in zip(recipe.query_lengths, recipe.key_lengths))
        result = BackendResult(restored.reshape(query.shape), 'resident_grouped_fa2',
            (time.perf_counter()-started)*1000, pairs, pairs, 0, recipe.route_sha)
        # Additional accounting remains explicit and separate from KV payload.
        result.metadata_H2D_bytes = metadata_H2D_bytes
        result.resident_metadata_bytes = self.GPU_metadata_bytes
        return result
