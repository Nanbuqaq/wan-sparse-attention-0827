"""Batched physical raw-cache composition; never chooses logical Attention edges.

CPU request compilation is vectorized. Cache entries own their GPU allocations:
evicting one entry cannot leave a whole staging slab invisibly resident. Temporary
packing/gather allocations and index H2D are reported separately from KV payload.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from .history_cache import CachedRawHistoryBlock, RawHistoryBlockCacheKey
from .rope import build_sparse_positions, apply_selected_rope


@dataclass(frozen=True)
class RawRequestPlan:
    route_sha: str
    shape: tuple[int, int, int]  # batch, heads, union width
    frame_tokens: int
    block_tokens: int
    blocks: tuple[tuple[int, int, int, int, int], ...]  # batch, head, frame, start, end
    source_indices: torch.Tensor
    destination_indices: torch.Tensor
    requested_counts: tuple[int, ...]


def compile_raw_requests(route, *, frame_tokens, block_tokens):
    if frame_tokens < 1 or block_tokens < 1:
        raise ValueError('positive physical geometry required')
    frames = route.union_frame_ids.detach().cpu().numpy()
    tokens = route.union_token_ids.detach().cpu().numpy()
    if frames.ndim != 3 or frames.shape != tokens.shape:
        raise ValueError('matching batch/head/union coordinates required')
    batch, heads, width = frames.shape
    valid = frames >= 0
    if np.any((tokens[valid] < 0) | (tokens[valid] >= frame_tokens)):
        raise ValueError('invalid within-frame token coordinate')
    bi, hi, ui = np.nonzero(valid)
    fs, ts = frames[valid], tokens[valid]
    block_count = (frame_tokens+block_tokens-1)//block_tokens
    frame_base = int(fs.max())+1 if len(fs) else 1
    if batch*heads*frame_base*block_count >= 2**63:
        raise ValueError('coordinate encoding overflow')
    codes = ((bi*heads+hi)*frame_base+fs)*block_count+ts//block_tokens
    _, first, inverse = np.unique(codes, return_index=True, return_inverse=True)
    # Preserve the original dictionary's first-appearance order, including LRU.
    first_order = np.argsort(first, kind='stable')
    remap = np.empty_like(first_order)
    remap[first_order] = np.arange(len(first_order))
    ordinal = remap[inverse]
    first = first[first_order]
    starts = (ts[first]//block_tokens)*block_tokens
    ends = np.minimum(starts+block_tokens, frame_tokens)
    descriptors = tuple(zip(bi[first].tolist(), hi[first].tolist(), fs[first].tolist(), starts.tolist(), ends.tolist()))
    offsets = np.concatenate(([0], np.cumsum(ends-starts)))
    source = offsets[ordinal]+ts%block_tokens
    destination = (bi*width+ui)*heads+hi
    return RawRequestPlan(route.digest(), (batch, heads, width), frame_tokens, block_tokens, descriptors,
        torch.from_numpy(source.astype(np.int64)), torch.from_numpy(destination.astype(np.int64)),
        tuple(np.bincount(ordinal, minlength=len(descriptors)).tolist()))


def compose_raw_entries(entries, plan, *, dim, dtype, device):
    batch, heads, width = plan.shape
    key = torch.zeros((batch, width, heads, dim), dtype=dtype, device=device)
    value = torch.zeros_like(key)
    if len(entries) != len(plan.blocks):
        raise ValueError('one entry per compiled physical block required')
    if not entries:
        return key, value, 0, 0
    for entry, spec in zip(entries, plan.blocks):
        bi, hi, fi, start, end = spec
        if (entry.key.batch_id, entry.key.head_id, entry.key.frame_id,
                entry.key.token_start, entry.key.token_end) != spec:
            raise ValueError('cache entry ownership differs from compiled request')
        if entry.key_unrotated.shape != (end-start, dim) or entry.value.shape != (end-start, dim):
            raise ValueError('cache entry physical shape mismatch')
    source = plan.source_indices.to(device)
    destination = plan.destination_indices.to(device)
    # One gather and scatter per K/V, instead of two launches per selected token.
    packed = torch.cat([e.key_unrotated for e in entries], dim=0)
    key.view(-1, dim).index_copy_(0, destination, packed.index_select(0, source))
    del packed
    packed = torch.cat([e.value for e in entries], dim=0)
    value.view(-1, dim).index_copy_(0, destination, packed.index_select(0, source))
    index_bytes = (source.numel()+destination.numel())*8 if torch.device(device).type == 'cuda' else 0
    return key, value, index_bytes, 2 if index_bytes else 0


def materialize_raw_batched(archive, layer_id, route, cache, *, device, current_frame_id,
                            freqs, block_tokens=64, candidate_frame_ids=None, staging_pool=None):
    from .archive import MaterializedHistory
    started = time.perf_counter()
    device = torch.device(device)
    layer = archive._layers.get(int(layer_id), {})
    if not layer:
        raise KeyError(f'layer {layer_id} has no archived frames')
    frame_tokens = archive.spatial_height*archive.spatial_width
    # One CPU plan per layer, no raw KV references and no unbounded plan history.
    if not hasattr(archive, '_raw_request_plans'):
        archive._raw_request_plans = {}
    marker = (route.digest(), frame_tokens, block_tokens)
    prior = archive._raw_request_plans.get(int(layer_id))
    compiled = prior[1] if prior is not None and prior[0] == marker else compile_raw_requests(
        route, frame_tokens=frame_tokens, block_tokens=block_tokens)
    archive._raw_request_plans[int(layer_id)] = (marker, compiled)
    first = next(iter(layer.values()))
    dim, dtype = first.key.shape[-1], first.key.dtype
    entries, misses = [], []
    hit_bytes = 0
    for index, spec in enumerate(compiled.blocks):
        bi, hi, frame, start, end = spec
        if frame not in layer:
            raise KeyError(f'route frame {frame} is not archived in layer {layer_id}')
        if not (bi < layer[frame].key.shape[0] and hi < layer[frame].key.shape[2]):
            raise ValueError('raw request batch/head exceeds archived geometry')
        cache_key = RawHistoryBlockCacheKey(batch_id=bi, layer_id=int(layer_id), head_id=hi,
            archive_epoch=archive.epoch, frame_id=frame,
            frame_storage_version=archive.frame_storage_version(layer_id, frame),
            token_start=start, token_end=end, dtype=str(dtype), device=str(device))
        cached = cache.get(cache_key)
        entries.append(cached)
        if cached is None:
            misses.append((index, cache_key, spec))
        else:
            hit_bytes += cached.bytes
    prepare_s = time.perf_counter()-started
    lease = None
    pack_s = pin_s = h2d_s = cache_store_s = 0.
    transfer_bytes = payload_bytes = 0
    if misses:
        try:
            missing_tokens = sum(spec[4]-spec[3] for _, _, spec in misses)
            pack_started = time.perf_counter()
            if staging_pool is not None:
                lease = staging_pool.acquire((missing_tokens, dim), dtype, fused=False)
                host_k, host_v = lease.key, lease.value
            else:
                pinned = archive.config.pin_memory and device.type == 'cuda'
                host_k = torch.empty((missing_tokens, dim), dtype=dtype, pin_memory=pinned)
                host_v = torch.empty_like(host_k, pin_memory=pinned)
            pin_s = time.perf_counter()-pack_started
            slices, offset = [], 0
            for index, cache_key, (bi, hi, frame, start, end) in misses:
                count = end-start
                host_k[offset:offset+count].copy_(layer[frame].key[bi, start:end, hi])
                host_v[offset:offset+count].copy_(layer[frame].value[bi, start:end, hi])
                slices.append((index, cache_key, offset, count))
                offset += count
                payload_bytes += compiled.requested_counts[index]*2*dim*host_k.element_size()
            pack_s = time.perf_counter()-pack_started-pin_s
            copy_started = time.perf_counter()
            gpu_k = host_k.to(device, non_blocking=host_k.is_pinned())
            gpu_v = host_v.to(device, non_blocking=host_v.is_pinned())
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            h2d_s = time.perf_counter()-copy_started
            transfer_bytes = 2*missing_tokens*dim*host_k.element_size()
            store_started = time.perf_counter()
            for index, cache_key, offset, count in slices:
                # Owning clones: cache accounting never hides a retained slab.
                entry = CachedRawHistoryBlock(cache_key, gpu_k[offset:offset+count].clone(),
                                               gpu_v[offset:offset+count].clone())
                cache.put(entry)
                entries[index] = entry
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            cache_store_s = time.perf_counter()-store_started
            del gpu_k, gpu_v
        finally:
            if lease is not None:
                staging_pool.release(lease)
    restore_started = time.perf_counter()
    key_raw, value, index_bytes, index_copies = compose_raw_entries(entries, compiled, dim=dim,
                                                                  dtype=dtype, device=device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    restore_s = time.perf_counter()-restore_started
    frames, tokens = route.union_frame_ids, route.union_token_ids
    if candidate_frame_ids is None:
        candidate_frame_ids = list(dict.fromkeys(frames[frames >= 0].tolist()))
    positions = build_sparse_positions(frame_ids=frames.clamp_min(0), token_ids=tokens.clamp_min(0),
        current_frame_id=current_frame_id, spatial_width=archive.spatial_width,
        rope_policy=archive.config.rope_policy, max_relative_age=archive.config.max_relative_age,
        candidate_frame_ids=torch.as_tensor(candidate_frame_ids, dtype=torch.long).cpu())
    rope_started = time.perf_counter()
    key = key_raw if freqs is None else apply_selected_rope(key_raw, positions.to(device), freqs.to(device))
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    rope_s = time.perf_counter()-rope_started
    return MaterializedHistory(key_unrotated=key_raw, key=key, value=value, positions=positions,
        transferred_bytes=transfer_bytes, payload_bytes=payload_bytes,
        padding_bytes=max(0, transfer_bytes-payload_bytes), cpu_gather_s=pin_s+pack_s,
        cpu_prepare_s=prepare_s, cpu_pack_s=pack_s, cpu_allocate_pin_s=pin_s,
        h2d_s=h2d_s, h2d_copy_count=2 if misses and device.type == 'cuda' else 0,
        source_run_count=len(misses), gpu_restore_s=restore_s, rope_s=rope_s,
        materialize_total_s=time.perf_counter()-started, staging_mode='batched_raw_block64',
        staging_reused=bool(lease and lease.reused), cache_hit=not misses,
        cache_hit_bytes=hit_bytes, cache_miss_bytes=transfer_bytes,
        cache_store_s=cache_store_s, restore_index_h2d_bytes=index_bytes,
        restore_index_h2d_copy_count=index_copies)
