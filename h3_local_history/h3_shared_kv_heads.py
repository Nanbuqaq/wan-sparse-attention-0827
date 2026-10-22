"""Head sharding for Agent 5's unchanged FixedKVAttention implementation.

No mask, cache lifecycle or sampling rule is defined here. Native Q/K/V
projection outputs are split by head; only current attention outputs gather
before the native output projection. All ranks execute the same request.
"""
import torch
import torch.distributed as dist


def install_head_shards(blocks, group=None):
    if not dist.is_initialized():
        raise ValueError('Head-sharded shared KV requires an initialized same-video group')
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    handles = []
    for block in blocks:
        attn = block.attn
        if hasattr(attn, '_shared_kv_full_heads'):
            raise ValueError('Head sharding already installed')
        full_heads = attn.heads
        if full_heads % world:
            raise ValueError('The full head count must divide over the group')
        width = attn.to_q.out_features
        if width % full_heads:
            raise ValueError('Invalid native QKV layout')
        lo, hi = rank*(width//world), (rank+1)*(width//world)

        def shard(module, inputs, result, lo=lo, hi=hi):
            # Own the small shard: a view would retain the full V allocation.
            return result[..., lo:hi].contiguous()

        def gather(module, inputs):
            value = inputs[0].contiguous()
            pieces = [torch.empty_like(value) for _ in range(world)]
            dist.all_gather(pieces, value, group=group)
            return (torch.cat(pieces, dim=-1), *inputs[1:])

        for module in (attn.to_q, attn.to_k, attn.to_v):
            handles.append(module.register_forward_hook(shard))
        handles.append(attn.to_out[0].register_forward_pre_hook(gather))
        attn._shared_kv_full_heads = full_heads
        attn.heads = full_heads//world
    return handles


def require_same_request(identity, group=None):
    descriptors = [None]*dist.get_world_size(group)
    dist.all_gather_object(descriptors, identity, group=group)
    if any(value != identity for value in descriptors):
        raise ValueError('Independent cases/noise cannot share one head-sharded video group')
