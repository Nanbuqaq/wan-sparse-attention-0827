"""Diagnostic-only two-stage memory observations; never an online selector.

Fresh shadow routes consume current Q summaries and committed archive indices.
Their results are recorded AFTER the executed output, not fed back into it.
Coarse frame retrieval and fine per-head admission are distinct populations.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import torch

from .selectors import summarize_query_for_pretransfer


def coordinate_codes(plan, *, frame_base: int, token_base: int, block_size: int = 1):
    frames = plan.union_frame_ids.detach().cpu().long()
    tokens = plan.union_token_ids.detach().cpu().long()
    batch, heads, _ = frames.shape
    owner = torch.arange(batch * heads).reshape(batch, heads, 1).expand_as(frames)
    valid = (frames >= 0) & (tokens >= 0)
    if valid.any() and (int(frames[valid].max()) >= frame_base or int(tokens[valid].max()) >= token_base):
        raise ValueError('coordinate exceeds declared geometry')
    block_base = (token_base + block_size - 1) // block_size
    return torch.unique(((owner[valid] * frame_base + frames[valid]) * block_base
                         + tokens[valid] // block_size), sorted=True)


def compare_coordinates(left, right, *, token_base: int, block_size: int = 1):
    if left.union_frame_ids.shape[:2] != right.union_frame_ids.shape[:2]:
        raise ValueError('batch/head ownership mismatch')
    frame_base = max(int(left.union_frame_ids.max()), int(right.union_frame_ids.max()), 0) + 1
    a = coordinate_codes(left, frame_base=frame_base, token_base=token_base, block_size=block_size)
    b = coordinate_codes(right, frame_base=frame_base, token_base=token_base, block_size=block_size)
    union = torch.unique(torch.cat((a, b))).numel()
    intersection = a.numel() + b.numel() - union
    return {'left_count': a.numel(), 'right_count': b.numel(),
            'intersection': intersection, 'union': union,
            'jaccard': intersection / union if union else 1.0,
            'right_recall_by_left': intersection / b.numel() if b.numel() else 1.0}


def frame_lifecycle(records, *, sink_size: int, recent_exclude: int, chunk_frames: int):
    """Observed coarse retrieval, not causal importance or future-unneeded proof.

    The final observation right-censors reuses. A frame becoming eligible late
    cannot be called permanently cold because it has not yet been selected.
    """
    rows = sorted(records, key=lambda row: row['query_frame'])
    starts = [int(row['query_frame']) for row in rows]
    if len(starts) != len(set(starts)):
        raise ValueError('duplicate chunk: do not mix cases or denoising calls')
    accesses, exposure = defaultdict(list), Counter()
    adjacent = []
    previous = None
    for row in rows:
        start = int(row['query_frame'])
        raw = row['selected_global_frames']
        selected = [int(x) for x in (raw[0] if raw and isinstance(raw[0], list) else raw)]
        if len(set(selected)) != len(selected):
            raise ValueError('duplicate coarse frame')
        eligible = range(sink_size, sink_size + max(0, int(row['num_evicted']) - recent_exclude))
        if not set(selected).issubset(eligible):
            raise ValueError('retrieved frame outside causal eligible pool')
        for frame in eligible:
            exposure[frame] += 1
        for frame in selected:
            accesses[frame].append(start)
        current = set(selected)
        if previous is not None:
            intersection, union = len(previous & current), len(previous | current)
            adjacent.append(intersection / union if union else 1.)
        previous = current
    gaps, counts = [], []
    for frame, times in sorted(accesses.items()):
        gaps.extend((b - a) / chunk_frames for a, b in zip(times, times[1:]))
        counts.append({'frame': frame, 'accesses': len(times), 'eligible_observations': exposure[frame],
                       'query_frames': times, 'last_gap_is_right_censored': True})
    ranked = sorted((len(v) for v in accesses.values()), reverse=True)
    total = sum(ranked)
    return {'scope': 'coarse_frame_selection_only', 'observed_chunks': len(rows),
            'eligible_frames_seen': len(exposure), 'selected_unique_frames': len(accesses),
            'never_selected_within_observation': len(set(exposure) - set(accesses)),
            'right_censored': True, 'retrievals': total,
            'adjacent_frame_jaccard': adjacent, 'revisit_gap_chunks': gaps,
            'top6_access_share': sum(ranked[:6]) / total if total else None,
            'top12_access_share': sum(ranked[:12]) / total if total else None,
            'frames': counts, 'semantic_importance_claim': False}


class MemoryDynamicsObserver:
    def __init__(self, output: Path, *, layers, shadow_starts):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.layers = set(layers)
        self.shadow_starts = set(shadow_starts)
        self.records = []
        self.first = {}
        self.current_timestep = None

    def __call__(self, *, module, query, route_plan, candidate_frame_ids,
                 current_start, denoising_pass, route_was_reused):
        if module.layer_id not in self.layers:
            return
        shadow = current_start in self.shadow_starts
        if denoising_pass > 0 and not shadow:
            return
        if module.sparse_config.method != 'transfer_vaware_hybrid_history':
            raise ValueError('this diagnostic freezes legacy Final only')
        if module.system_config.group_selection_policy != 'legacy_exact_union':
            raise ValueError('shadow replay must include any group-policy transformation')
        before_sha = route_plan.digest()
        start = time.perf_counter()
        fresh = route_plan
        if shadow and route_was_reused:
            summary = summarize_query_for_pretransfer(query.detach(),
                int(module.sparse_config.method_params.get('query_block_size', 64)))
            fresh = module.history_archive.route_indexed(module.layer_id, summary,
                candidate_frame_ids, exact_k_tokens=route_plan.exact_k_tokens)
        shadow_s = time.perf_counter() - start
        marker = (module.layer_id, current_start)
        first = self.first.setdefault(marker, fresh)
        tokens_per_frame = module.history_archive.spatial_height * module.history_archive.spatial_width
        row = {'layer': module.layer_id, 'current_start': current_start,
               'denoising_pass': denoising_pass, 'timestep': self.current_timestep,
               'phase': 'clean_context_commit' if self.current_timestep == 0 else 'denoising',
               'route_was_reused': route_was_reused, 'fresh_shadow_computed': shadow and route_was_reused,
               'executed_route_sha': before_sha, 'fresh_route_sha': fresh.digest(),
               'candidate_frame_ids': candidate_frame_ids.detach().cpu().reshape(-1).tolist(),
               'selected_tokens': route_plan.unique_history_tokens,
               'candidate_tokens_per_head': route_plan.candidate_history_tokens,
               'global_executed_density': route_plan.global_executed_density,
               'shadow_diagnostic_s': shadow_s,
               'executed_vs_fresh_tokens': compare_coordinates(route_plan, fresh, token_base=tokens_per_frame),
               'executed_vs_fresh_blocks': compare_coordinates(route_plan, fresh, token_base=tokens_per_frame, block_size=64),
               'first_vs_fresh_tokens': compare_coordinates(first, fresh, token_base=tokens_per_frame),
               'executor_storage': route_plan.grouped_executor_storage(head_dim=query.shape[-1], element_size=query.element_size())}
        if route_plan.digest() != before_sha:
            raise RuntimeError('diagnostic mutated executed route')
        # Coordinates preserve head ownership; never label this KV sharing across layers.
        payload = {'record': row, 'executed_route': route_plan.state_dict(), 'fresh_route': fresh.state_dict()}
        path = self.output / f'layer{module.layer_id:02d}_start{current_start:08d}_pass{denoising_pass:02d}.pt'
        if path.exists():
            raise FileExistsError(path)
        torch.save(payload, path)
        self.records.append(row)

    def finish(self):
        payload = {'scope': 'post_output_shadow_diagnostic_only', 'teacher_used': False,
                   'shadow_routes_executed': False, 'end_to_end_speed_claim': False,
                   'records': self.records}
        data = json.dumps(payload, indent=2, sort_keys=True) + '\n'
        (self.output / 'observations.json').write_text(data)
        return hashlib.sha256(data.encode()).hexdigest()
