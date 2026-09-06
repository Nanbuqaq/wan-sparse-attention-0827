"""Route-proven rectangular head batching, without varlen replication/scatter."""
import time
import torch

from .profiling import profiled


def require_rectangular_shared_union(plan, *, query_shape, exact_tokens, union_tokens):
    batch, queries, heads, _ = query_shape
    if plan.query_labels.shape != (batch, heads, queries) or plan.groups != 1:
        raise ValueError('batched FA2 requires exactly one query group per head')
    if plan.union_frame_ids.shape != (batch, heads, union_tokens) or plan.exact_k_tokens != exact_tokens:
        raise ValueError('batched FA2 geometry differs from the route')
    if bool((plan.query_labels != 0).any()) or bool((plan.query_group_sizes != queries).any()):
        raise ValueError('batched FA2 requires all query rows in the shared group')
    if bool((plan.group_history_counts != union_tokens).any()) or bool((plan.union_frame_ids < 0).any()):
        raise ValueError('batched FA2 does not allow padded or partially consumed unions')
    canonical = torch.arange(union_tokens, device=plan.group_union_indices.device)
    if not torch.equal(plan.group_union_indices[..., :union_tokens], canonical.view(1, 1, 1, -1).expand(batch, heads, 1, -1)):
        raise ValueError('batched FA2 must preserve the original per-head KV order')


@profiled('attention/route_proven_batched_fa2_complete')
def execute_rectangular_shared_fa2(query, exact_key, exact_value, history_key, history_value, plan):
    if not query.is_cuda:
        raise RuntimeError('real CUDA FA2 required; no dense fallback')
    import flash_attn
    from .backends import BackendResult
    started = time.perf_counter()
    require_rectangular_shared_union(plan, query_shape=query.shape, exact_tokens=exact_key.shape[1], union_tokens=history_key.shape[1])
    for key, value in ((exact_key, exact_value), (history_key, history_value)):
        if key.shape != value.shape or key.shape[0] != query.shape[0] or key.shape[2:] != query.shape[2:]:
            raise ValueError('batched FA2 Q/K/V dimensions differ')
        if key.dtype != query.dtype or value.dtype != query.dtype or key.device != query.device or value.device != query.device:
            raise ValueError('batched FA2 Q/K/V dtype/device differ')
    key = torch.cat((exact_key, history_key), dim=1)
    value = torch.cat((exact_value, history_value), dim=1)
    output = flash_attn.flash_attn_func(query, key, value, dropout_p=0., causal=False)
    torch.cuda.synchronize(query.device)
    elapsed = (time.perf_counter()-started)*1000
    pairs = query.shape[0]*query.shape[2]*query.shape[1]*key.shape[1]
    return BackendResult(output, 'batched_fa2', elapsed, pairs, pairs, 0, plan.digest())
