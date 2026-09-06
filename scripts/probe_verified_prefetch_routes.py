#!/usr/bin/env python3
"""Causal route-only prefetch screening, with exact partial-Block64 byte counts.

This is a prediction/physical-traffic analysis, not measured overlap or speed.
All predictions are made before inspecting target logical coordinates. Layer
coordinates are translated into the target layer's storage domain; KV tensors
are never shared across layers. Only first calls (cache misses) are sampled.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, OrderedDict
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.route_plan import HistoryRoutePlan

TOKENS = 1560
TOKEN_BYTES = 512  # BF16 K+V at head_dim128; ownership includes the head.


def blocks(plan):
    frames, tokens = plan.union_frame_ids, plan.union_token_ids
    if frames.shape[0] != 1:
        raise ValueError('recorded development analysis is batch-one')
    result = set()
    for head in range(frames.shape[1]):
        valid = frames[0, head] >= 0
        if bool(((tokens[0, head, valid] < 0) | (tokens[0, head, valid] >= TOKENS)).any()):
            raise ValueError('invalid within-frame token')
        codes = torch.unique(frames[0, head, valid]*25+tokens[0, head, valid]//64)
        result.update((head, int(c)//25, int(c)%25) for c in codes)
    return result


def width(block):
    if not 0 <= block[2] < 25:
        raise ValueError('invalid within-frame Block64')
    return min(64, TOKENS-block[2]*64)


def byte_count(selected):
    return sum(width(block)*TOKEN_BYTES for block in selected)


def admit(predicted, candidates, heads, *, fraction=.25):
    """Budget comes from causal coarse candidates, never target actual count."""
    selected, used = set(), defaultdict(int)
    allowed = set(candidates)
    budget = int(len(allowed)*TOKENS*fraction)
    for block in predicted:
        head, frame, _ = block
        if not 0 <= head < heads or frame not in allowed or block in selected:
            continue
        if used[head]+width(block) > budget:
            continue
        selected.add(block)
        used[head] += width(block)
    return selected


def physical_metrics(predicted, actual, logical_bytes):
    hits, misses, extras = predicted & actual, actual-predicted, predicted-actual
    actual_bytes = byte_count(actual)
    predicted_bytes = byte_count(predicted)
    return {'actual_whole_block_bytes': actual_bytes, 'exact_logical_payload_bytes': logical_bytes,
        'predicted_bytes': predicted_bytes, 'hit_bytes': byte_count(hits),
        'miss_bytes': byte_count(misses), 'extra_bytes': byte_count(extras),
        'byte_recall': byte_count(hits)/actual_bytes if actual_bytes else 1.,
        'byte_precision': byte_count(hits)/predicted_bytes if predicted_bytes else None,
        'total_prefetch_plus_completion_bytes': predicted_bytes+byte_count(misses),
        'traffic_ratio_vs_whole_block': (predicted_bytes+byte_count(misses))/actual_bytes if actual_bytes else 1.,
        'traffic_ratio_vs_exact': (predicted_bytes+byte_count(misses))/logical_bytes if logical_bytes else 1.,
        'final_block_coverage_exact_actual_after_discarding_extras': (hits | misses) == actual,
        'timeliness': None, 'exposed_wait_s': None, 'measured_H2D_overlap': False}


def summarize(rows):
    total = sum(r['actual_whole_block_bytes'] for r in rows)
    prediction = sum(r['predicted_bytes'] for r in rows)
    hits = sum(r['hit_bytes'] for r in rows)
    return {'calls': len(rows), 'aggregate_byte_recall': hits/total,
        'aggregate_byte_precision': hits/prediction if prediction else None,
        'aggregate_extra_bytes': sum(r['extra_bytes'] for r in rows),
        'aggregate_traffic_ratio_vs_whole_block': sum(r['total_prefetch_plus_completion_bytes'] for r in rows)/total,
        'median_byte_recall': statistics.median(r['byte_recall'] for r in rows),
        'causal_prefetch_budget': 'at most25% candidate tokens per head before target route'}


def lru_replay(requests, budget_bytes):
    """Persistent raw occupancy only; excludes the separate assembled union.

    Query all current hits before inserting misses, matching the existing raw
    cache's get-then-put ordering. No future request influences replacement.
    """
    cache, used, peak = OrderedDict(), 0, 0
    records = []
    for start, requested in requests:
        hit, missing = set(), []
        for block in sorted(requested):
            if block in cache:
                hit.add(block)
                cache.move_to_end(block)
            else:
                missing.append(block)
        for block in missing:
            size = width(block)*TOKEN_BYTES
            if size > budget_bytes:
                continue
            while used+size > budget_bytes:
                _, removed = cache.popitem(last=False)
                used -= removed
            cache[block] = size
            used += size
            peak = max(peak, used)
        records.append({'current_start': start, 'hit_bytes': byte_count(hit),
            'missing_bytes': byte_count(set(missing)), 'requested_bytes': byte_count(requested),
            'persistent_cache_bytes': used})
    total = sum(r['requested_bytes'] for r in records)
    return {'budget_bytes': budget_bytes, 'peak_persistent_bytes': peak,
        'first_pass_requests': len(records), 'aggregate_byte_hit_rate': sum(r['hit_bytes'] for r in records)/total,
        'miss_bytes': sum(r['missing_bytes'] for r in records), 'records': records,
        'same_chunk_five_call_reuse_excluded': True, 'time_saved_not_predicted': True}


def analyze(root):
    by_chunk, previous = defaultdict(dict), {}
    evidence = []
    for path in sorted((root/'route_observations').glob('layer*_pass00.pt')):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        record = payload['record']
        plan = HistoryRoutePlan.from_state_dict(payload['executed_route'])
        if record['denoising_pass'] != 0 or plan.digest() != record['executed_route_sha']:
            raise ValueError('first-call executed route audit mismatch')
        by_chunk[record['current_start']][record['layer']] = (record, plan)
        evidence.append({'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    rows = []
    for start, layers in sorted(by_chunk.items()):
        for layer, (record, plan) in sorted(layers.items()):
            candidates = record['candidate_frame_ids']
            heads = plan.union_frame_ids.shape[1]
            sources = {}
            if layer-1 in layers:
                sources['previous_adjacent_layer'] = blocks(layers[layer-1][1])
            if layer in previous:
                sources['same_layer_previous_chunk'] = previous[layer]
            universe = [(h, f, b) for h in range(heads) for f in sorted(candidates) for b in range(25)]
            random.Random(20260906+start*31+layer).shuffle(universe)
            predictions = {'random_causal_control': admit(universe, candidates, heads)}
            for name, source in sources.items():
                predictions[name] = admit(sorted(source), candidates, heads)
            # Actual target coordinates first enter metrics here, never admission.
            actual = blocks(plan)
            for name, prediction in predictions.items():
                rows.append({'current_start': start, 'target_layer': layer, 'predictor': name,
                    'target_route_sha256': plan.digest(), 'candidate_frames': candidates,
                    'target_layer_owns_all_copied_KV': True,
                    **physical_metrics(prediction, actual, plan.unique_history_tokens*TOKEN_BYTES)})
            # Update completed source chunk only AFTER all target predictions.
        previous = {layer: blocks(plan) for layer, (_, plan) in layers.items()}
    groups = defaultdict(list)
    for row in rows:
        groups[row['predictor']].append(row)
    # Random baseline is additionally matched to each predictor's target calls.
    random_by_call = {(r['current_start'], r['target_layer']): r for r in groups['random_causal_control']}
    summary = {k: summarize(v) for k, v in groups.items()}
    for name in ('previous_adjacent_layer', 'same_layer_previous_chunk'):
        if name in groups:
            summary[name]['random_same_calls'] = summarize([random_by_call[r['current_start'], r['target_layer']]
                                                           for r in groups[name]])
    layer_requests = defaultdict(list)
    for start, layers in sorted(by_chunk.items()):
        for layer, (_, plan) in sorted(layers.items()):
            layer_requests[layer].append((start, blocks(plan)))
    residency = {str(layer): {str(mib): lru_replay(requests, mib*1024**2) for mib in (8, 16, 32, 64, 128)}
                 for layer, requests in layer_requests.items()}
    return {'records': rows, 'summary': summary, 'input_evidence': evidence,
        'per_layer_raw_residency': residency,
        'raw_cache_limit': 'six sampled layers separately; do not multiply observed hit rate into unsampled layers'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.workspace).resolve()/'results/metrics/memory_dynamics_d6b20e4'
    cohorts = {kind: analyze(root/f'{kind}120') for kind in ('motion', 'state')}
    result = {'status': 'pass', 'scope': 'offline_causal_prediction_physical_traffic_not_GPU_execution',
        'cohorts': cohorts, 'only_first_denoising_calls': True, 'only_true_adjacent_layer_prediction': True,
        'whole_Block64_tail_tokens': 24, 'payload_dtype': 'BF16', 'head_dim': 128,
        'physical_layout_does_not_change_target_logical_route': True,
        'cross_layer_KV_sharing': False, 'formal_promotion': False,
        'q_to_next_proto': 'not_evaluated; observer has coordinates, not consecutive-layer Q/prototypes',
        'limits': ['two finite development trajectories', 'no copy or compute timeline',
                   'physical block completeness differs from exact partial-token materialization',
                   'no claim about bounded total archive or net speedup']}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v['summary'] for k, v in cohorts.items()}, indent=2))


if __name__ == '__main__':
    main()
