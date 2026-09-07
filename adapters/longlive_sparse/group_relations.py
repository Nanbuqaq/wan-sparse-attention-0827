"""Exploratory query groups and conditional history admission.

The selector accepts only OnlineRoutingContext and coordinate metadata. This
is not yet an identity/scene/state semantic classifier or a promoted method.
"""
import math
import time

import torch
import torch.nn.functional as F

from .ar_routing import build_route_plan
from .contexts import OnlineRoutingContext
from .selectors import PretransferQuerySummary
from .utility import compute_online_utility_proxy
from .profiling import synchronize_cuda


def summarize_groups(query, *, grouping, spatial_height, spatial_width, seed=20260907):
    if query.ndim != 4 or query.shape[1] % (spatial_height*spatial_width):
        raise ValueError('query must contain complete spatial frames')
    if grouping not in ('random_balanced', 'spatial_quadrants', 'query_features'):
        raise ValueError('unknown grouping')
    if query.is_cuda:
        synchronize_cuda(query.device)
    begin = time.perf_counter()
    batch, count, heads, dim = query.shape
    groups = 4
    values = query.permute(0, 2, 1, 3).float()
    if grouping == 'spatial_quadrants':
        token = torch.arange(count, device=query.device) % (spatial_height*spatial_width)
        labels = ((token//spatial_width >= spatial_height//2).long()*2
                  + (token % spatial_width >= spatial_width//2).long())
        labels = labels.view(1, 1, -1).expand(batch, heads, -1)
    elif grouping == 'random_balanced':
        generator = torch.Generator(device='cpu').manual_seed(seed)
        labels = torch.stack([torch.randperm(count, generator=generator) % groups for _ in range(batch*heads)])
        labels = labels.reshape(batch, heads, count).to(query.device)
    else:
        # Four fixed-iteration spherical clusters of current Q only. No future
        # output, historical full K/V or teacher weights enter grouping.
        normalized = F.normalize(values, dim=-1)
        indices = torch.linspace(0, count-1, groups, device=query.device).long()
        centers = normalized[:, :, indices].clone()
        for _ in range(4):
            labels = torch.einsum('bhqd,bhgd->bhqg', normalized, centers).argmax(-1)
            parts = []
            for group in range(groups):
                mask = (labels == group).float()
                size = mask.sum(-1, keepdim=True)
                mean = torch.einsum('bhq,bhqd->bhd', mask, normalized)/size.clamp_min(1)
                parts.append(torch.where(size > 0, F.normalize(mean, dim=-1), centers[:, :, group]))
            centers = torch.stack(parts, dim=2)
    centroids, counts = [], []
    for group in range(groups):
        mask = (labels == group).float()
        size = mask.sum(-1)
        centroids.append(torch.einsum('bhq,bhqd->bhd', mask, values)/size.clamp_min(1)[..., None])
        counts.append(size.long())
    centroids = torch.stack(centroids, dim=2)
    counts = torch.stack(counts, dim=2)
    if query.is_cuda:
        synchronize_cuda(query.device)
    compute_s = time.perf_counter()-begin
    begin = time.perf_counter()
    labels, centroids, counts = labels.cpu(), centroids.cpu(), counts.cpu()
    d2h_s = time.perf_counter()-begin
    return PretransferQuerySummary(query_labels=labels, query_centroids=centroids,
        query_group_sizes=counts, query_tokens=count, block_size=0,
        summary_bytes=sum(x.numel()*x.element_size() for x in (labels, centroids, counts)),
        q_summary_s=compute_s, d2h_s=d2h_s)


def build_group_relation_route(context: OnlineRoutingContext, query_labels, frame_ids, token_ids, *,
                               exact_tokens, grouping, admission, density=.25, recency_half_life=None,
                               selection_impl='block_expand'):
    if admission not in ('shared', 'per_group') or not 0 < density <= 1:
        raise ValueError('invalid relation admission or density')
    if frame_ids.shape != token_ids.shape or frame_ids.shape[:2] != query_labels.shape[:2]:
        raise ValueError('coordinate/query geometry mismatch')
    if query_labels.device.type != 'cpu' or frame_ids.device.type != 'cpu':
        raise ValueError('this selector consumes CPU coordinate metadata')
    prior = None
    if recency_half_life is not None:
        if recency_half_life <= 0:
            raise ValueError('recency half-life must be positive')
        prior = -math.log(2)*context.block_age/recency_half_life
    probability = compute_online_utility_proxy(context, log_prior=prior).block_probabilities
    batch, heads, groups, _ = probability.shape
    total = frame_ids.shape[-1]
    budget = max(1, min(total, int(round(density*total))))
    if selection_impl not in ('block_expand', 'token_reference'):
        raise ValueError('unknown equivalent selection implementation')
    base = int(max(context.block_token_ends.max(), token_ids.max()+1))
    block_codes = context.block_frame_ids*base + context.block_token_starts
    ends = context.block_frame_ids*base + context.block_token_ends
    ordered_codes, code_order = block_codes.sort()
    if bool((ends[code_order][:-1] > ordered_codes[1:]).any()):
        raise ValueError('overlapping prototype blocks')
    token_codes = frame_ids*base+token_ids
    offsets = torch.searchsorted(ordered_codes, token_codes.contiguous(), right=True)-1
    membership = code_order[offsets.clamp_min(0)]
    valid = ((offsets >= 0) & (frame_ids == context.block_frame_ids[membership]) &
             (token_ids >= context.block_token_starts[membership]) & (token_ids < context.block_token_ends[membership]))
    if not bool(valid.all()):
        raise ValueError('uncovered candidate token')
    widths = context.block_token_ends-context.block_token_starts
    canonical = torch.repeat_interleave(torch.arange(context.blocks), widths)
    block_table = None
    if (selection_impl == 'block_expand' and canonical.numel() == total and
            torch.equal(membership, canonical.view(1, 1, -1).expand_as(membership))):
        # Exact certification of the ordinary full-frame candidate layout.
        # Rank B block scores rather than sorting T repeated token scores.
        begin = torch.cat((widths.new_zeros(1), widths.cumsum(0)[:-1]))
        offset = torch.arange(int(widths.max())).view(1, -1)
        block_table = torch.where(offset < widths[:, None], begin[:, None]+offset, -1)
    selections = []
    compact_labels = query_labels.clone()
    for b in range(batch):
        head_selections = []
        for h in range(heads):
            scores = probability[b, h]
            if admission == 'shared':
                weights = context.query_group_sizes[b, h].float()
                weights = weights/weights.sum().clamp_min(1)
                scores = (scores*weights[:, None]).sum(0, keepdim=True).expand_as(scores)
            choices = []
            active_groups = torch.nonzero(context.query_group_sizes[b, h] > 0, as_tuple=False).flatten().tolist()
            for new_group, group in enumerate(active_groups):
                compact_labels[b, h][query_labels[b, h] == group] = new_group
                if block_table is not None:
                    order = torch.argsort(scores[group], descending=True, stable=True)
                    expanded = block_table.index_select(0, order).reshape(-1)
                    chosen = expanded[expanded >= 0][:budget]
                else:
                    per_token = scores[group].index_select(0, membership[b, h])
                    chosen = torch.argsort(per_token, descending=True, stable=True)[:budget]
                choices.append(chosen.sort().values)
            head_selections.append(choices)
        selections.append(head_selections)
    route = build_route_plan(method='group_relation_probe', routing_stage='pre-transfer',
        query_labels=compact_labels, selections=selections, history_frame_ids=frame_ids,
        history_token_ids=token_ids, candidate_history_tokens=total, exact_k_tokens=exact_tokens,
        density=density, metadata={'routing_identity': {'grouping': grouping, 'admission': admission,
            'recency_half_life': recency_half_life, 'prototype_space': 'unrotated'},
            'teacher_used': False, 'global_union_cap': None if admission == 'per_group' else density,
            'per_query_token_budget': budget,
            'granularity': 'Block64_priority_with_token_trimmed_budget_boundary',
            'selection_implementation': 'certified_block_expand' if block_table is not None else 'token_reference',
            'semantic_identity_state_classifier': False})
    expected_pairs = batch*heads*query_labels.shape[-1]*budget
    if route.history_pairs != expected_pairs:
        raise RuntimeError('grouping changed the declared per-query pair budget')
    return route
