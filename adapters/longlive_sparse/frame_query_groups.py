"""Whole-frame history selection for fixed spatial query groups.

Shared and split controls use identical causal frame prototypes and scoring.
Every selected KV is original, head-specific and already resident. Grouped
execution packs per-group/head sequences; this reference charges that duplication.
"""
import hashlib
import math
import time

import torch

from .query_balanced_value import normalized_values, stratified_sites

POLICIES = ('shared', 'split_shared', 'split_specific')


def spatial_query_geometry(frames, height, width):
    if frames < 1 or height < 4 or width < 4 or height % 2 or width % 2:
        raise ValueError('fixed four groups require an even spatial grid >=4')
    positions = torch.arange(frames * height * width)
    y = positions.remainder(height * width) // width
    x = positions.remainder(width)
    labels = (y >= height // 2).long() * 2 + (x >= width // 2).long()
    groups = torch.stack([positions[labels == g] for g in range(4)])
    sites = stratified_sites(frames, height, width)
    sample_groups = torch.stack([torch.where(labels[sites] == g)[0] for g in range(4)])
    return groups, sites, sample_groups


def whole_frame_prototypes(summaries, frame_tokens):
    """Exact weighted means of the existing past-clean Block64 summaries."""
    if not summaries:
        raise ValueError('empty optional history')
    keys, values = [], []
    for km, vm, counts in summaries:
        weights = counts.float()[:, None, None] / frame_tokens
        keys.append((km.float() * weights).sum(0))
        values.append((vm.float() * weights).sum(0))
    return torch.stack(keys), torch.stack(values)


def choose_frames(query, key_mean, value_mean, sites, sample_groups, budget, policy):
    if policy not in POLICIES or not 0 < budget <= key_mean.shape[0]:
        raise ValueError('unknown policy or invalid whole-frame budget')
    counts = torch.ones(key_mean.shape[0], device=query.device)
    # Equal whole-frame counts cancel in the softmax. Score matches the existing
    # normalized p*||Vmean|| control, with the selection granularity made explicit.
    a = normalized_values(query[sites], key_mean, value_mean, counts)
    shared_scores = a.sum(1)
    shared = shared_scores.argsort(dim=-1, descending=True, stable=True)[:, :budget]
    if policy == 'split_specific':
        scores = torch.stack([a.index_select(1, s).sum(1) for s in sample_groups])
        chosen = scores.argsort(dim=-1, descending=True, stable=True)[..., :budget]
    else:
        chosen = shared[None].expand(1 if policy == 'shared' else 4, -1, -1)
    return chosen, shared


def execute_frame_routes(q, k, v, frame_ids, query_ids, inverse_queries, cuq, cuk, offsets, timing=None):
    """One native FA2 varlen call; exact fixed-size packing, no dynamic nonzero."""
    from flash_attn import flash_attn_varlen_func
    if q.shape[0] != 1 or k.shape != v.shape or q.dtype != torch.bfloat16:
        raise ValueError('one original BF16 window required')
    groups, heads, frames = frame_ids.shape
    length = query_ids.shape[1]
    dim = q.shape[-1]
    selected_tokens = frames * offsets.numel()
    if timing is not None: timing[0].record()
    indices = (frame_ids[..., None] * offsets.numel() + offsets).flatten(-2)
    h = torch.arange(heads, device=q.device)[None, :, None]
    hk = k[0].permute(1, 0, 2)[h, indices].reshape(-1, 1, dim)
    hv = v[0].permute(1, 0, 2)[h, indices].reshape(-1, 1, dim)
    hq = q[0, query_ids].permute(0, 2, 1, 3).contiguous().reshape(-1, 1, dim)
    if timing is not None: timing[1].record()
    out = flash_attn_varlen_func(hq, hk, hv, cuq, cuk, length, selected_tokens,
                               dropout_p=0., causal=False)
    if timing is not None: timing[2].record()
    reordered = out.reshape(groups, heads, length, dim).permute(0, 2, 1, 3).reshape(-1, heads, dim)
    result = reordered.index_select(0, inverse_queries)[None].contiguous()
    if timing is not None: timing[3].record()
    return result


class FrameQueryRouter:
    def __init__(self, policy, token_grid, fraction=.5, pack_backend='torch'):
        if policy not in POLICIES or fraction != .5:
            raise ValueError('registered first pilot fixes policy and optional half budget')
        self.policy, self.grid, self.fraction = policy, tuple(token_grid), fraction
        if pack_backend not in ('torch','fused'):raise ValueError('unknown explicit route packing backend')
        self.pack_backend=pack_backend
        self.geometry = {}
        self.builds = self.hits = self.index_bytes = 0
        self.pending = []
        self.pending_bytes = self.peak_pending_bytes = 0
        self.audit_bytes = 0
        self.route_hasher = hashlib.sha256()
        self.prototype_cache = {}
        self.prototype_builds = self.prototype_hits = self.prototype_peak_bytes = 0

    def prepare(self, q, k, eligible, protected, summaries, frame_tokens, prototype_key=None):
        began = time.perf_counter()
        if math.prod(self.grid) != frame_tokens or q.shape[1] % frame_tokens:
            raise ValueError('actual 5B query geometry differs')
        candidate_frames = len(eligible)
        budget = int(candidate_frames * self.fraction)
        if candidate_frames % 2 or budget < 1:
            raise ValueError('pilot requires an exact nonempty half-frame budget')
        groups = 1 if self.policy == 'shared' else 4
        key = (q.shape[1], q.shape[2], k.shape[1], tuple(eligible), tuple(protected), str(q.device))
        meta = self.geometry.get(key)
        if meta is None:
            ids, sites, sample_groups = spatial_query_geometry(q.shape[1] // frame_tokens, *self.grid)
            if groups == 1:
                ids = torch.arange(q.shape[1])[None]
            visible = (len(protected) + budget) * frame_tokens
            tensors = dict(query_ids=ids, inverse_queries=ids.flatten().argsort(), sites=sites,
                           sample_groups=sample_groups, optional=torch.tensor(eligible, dtype=torch.long),
                           protected=torch.tensor(protected, dtype=torch.long), offsets=torch.arange(frame_tokens),
                           cuq=torch.arange(groups*q.shape[2]+1, dtype=torch.int32)*ids.shape[1],
                           cuk=torch.arange(groups*q.shape[2]+1, dtype=torch.int32)*visible)
            self.index_bytes += sum(t.numel()*t.element_size() for t in tensors.values())
            meta = {name:t.to(q.device) for name,t in tensors.items()}
            if len(self.geometry) >= 32:
                raise RuntimeError('query geometry capacity32 reached')
            self.geometry[key] = meta
            self.builds += 1
        else:
            self.hits += 1
        cached = None if prototype_key is None else self.prototype_cache.get(prototype_key[0])
        if cached is not None and cached[0] == prototype_key:
            _, km, vm = cached
            self.prototype_hits += 1
        else:
            km, vm = whole_frame_prototypes(summaries, frame_tokens)
            self.prototype_builds += 1
            if prototype_key is not None:
                if not 0 <= prototype_key[0] < 30: raise ValueError('5B layer bound exceeded')
                self.prototype_cache[prototype_key[0]] = (prototype_key, km, vm)
                self.prototype_peak_bytes = max(self.prototype_peak_bytes,
                    sum(t.numel()*t.element_size() for c in self.prototype_cache.values() for t in c[1:]))
        chosen, shared = choose_frames(q[0], km, vm, meta['sites'], meta['sample_groups'], budget, self.policy)
        selected = meta['optional'][chosen]
        mandatory = meta['protected'][None, None].expand(groups, q.shape[2], -1)
        frames = torch.cat([mandatory, selected], -1).sort(-1).values
        mask = torch.zeros((groups, q.shape[2], candidate_frames), device=q.device, dtype=torch.bool).scatter_(-1, chosen, True)
        shared_mask = torch.zeros((q.shape[2], candidate_frames), device=q.device, dtype=torch.bool).scatter_(-1, shared, True)
        # Small deferred metrics; never feed output/teacher back into selection.
        statistics = torch.stack([mask.any(0).any(0).sum()*frame_tokens+len(protected)*frame_tokens,
            mask.any(0).sum()*frame_tokens+q.shape[2]*len(protected)*frame_tokens,
            (mask != shared_mask[None]).sum(), mask.any(0).sum(), mask.sum()])
        return dict(meta=meta, frame_ids=frames, mask=mask, statistics=statistics,
                    budget=budget*frame_tokens, groups=groups,
                    prepare_host_s=time.perf_counter()-began,
                    packed_KV_bytes=2*frames.numel()*frame_tokens*q.shape[-1]*k.element_size(),
                    query_pack_bytes=q.numel()*q.element_size(),
                    route_index_temporary_bytes=frames.numel()*frame_tokens*8 if self.pack_backend=='torch' else 0,
                    prototype_temporary_bytes=(km.numel()+vm.numel())*4)

    def execute(self, q, k, v, plan, timing=None):
        if self.pack_backend=='fused':
            from .fused_frame_routes import execute_fused_frame_routes
            return execute_fused_frame_routes(q,k,v,plan,timing=timing)
        m = plan['meta']
        return execute_frame_routes(q, k, v, plan['frame_ids'], m['query_ids'], m['inverse_queries'],
                                    m['cuq'], m['cuk'], m['offsets'], timing)

    def record(self, row, plan):
        mask = plan['mask'].detach() if row['layer'] == 14 else None
        tensors = [plan['statistics']] + ([] if mask is None else [mask])
        self.pending.append((row, plan['statistics'], mask))
        self.pending_bytes += sum(x.numel()*x.element_size() for x in tensors)
        self.peak_pending_bytes = max(self.peak_pending_bytes, self.pending_bytes)
        if self.pending_bytes > 8*1024**2:
            raise RuntimeError('query audit capacity8MiB reached')

    def audit(self):
        if self.pending:
            statistics = torch.stack([x[1] for x in self.pending]).cpu().tolist()
            self.audit_bytes += len(statistics)*5*8
            for (row, _, mask), s in zip(self.pending, statistics):
                row.update(physical_token_union=s[0], per_head_union_token_sum=s[1],
                    changed_optional_group_head_frames_from_shared=s[2],
                    optional_union_head_frames=s[3], optional_scheduled_group_head_frames=s[4])
                if mask is not None:
                    raw = mask.cpu().numpy().tobytes()
                    self.audit_bytes += len(raw)
                    self.route_hasher.update(str((row['call'],row['layer'],row['current_frame'])).encode())
                    self.route_hasher.update(raw)
            self.pending.clear()
            self.pending_bytes = 0
        return dict(policy=self.policy, pack_backend=self.pack_backend, granularity='whole_frame', geometry_builds=self.builds,
            geometry_hits=self.hits, index_H2D_bytes=self.index_bytes,
            prototype_builds=self.prototype_builds, prototype_hits=self.prototype_hits,
            prototype_cache_GPU_peak_bytes=self.prototype_peak_bytes,
            geometry_GPU_owned_bytes=sum(t.numel()*t.element_size() for m in self.geometry.values() for t in m.values()),
            deferred_audit_GPU_peak_bytes=self.peak_pending_bytes, audit_D2H_bytes=self.audit_bytes,
            layer14_route_sha256=self.route_hasher.hexdigest(),
            source='past committed clean frame means; current sampled Q; original resident raw KV',
            groups=1 if self.policy=='shared' else 4, archive_H2D_saving_claim=False,
            clean_and_first_cut='unchanged full native graph', native_context_update_unchanged=True,
            cost_scope='host spans include GPU submission; gather/FA2 combined unless explicitly replay-profiled')
