#!/usr/bin/env python3
"""Optimistic Q-role diagnostic on the exact Dense actor, not online runtime.

Past masks come from the offline Dense/SAM2 teacher. Current/future masks are
isolated evaluation labels. Full Q/K are reduced on CPU for this offline probe;
the role function receives summaries only. No learned temperature or tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.memory_roles import causal_query_identity_probability
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def group_means(tokens, block=64):
    if tokens.ndim != 1 or tokens.numel() == 0:
        raise ValueError('nonempty one-dimensional tokens required')
    return torch.stack([tokens[i:i+block].float().mean() for i in range(0, len(tokens), block)])


def role_metrics(prediction, label, weights):
    if prediction.shape != label.shape or weights.shape != label.shape:
        raise ValueError('shared group shape required')
    if not all(bool(torch.isfinite(x).all()) for x in (prediction, label, weights)):
        raise ValueError('finite probabilities and weights required')
    if bool((weights <= 0).any()) or bool(((prediction < 0) | (prediction > 1) | (label < 0) | (label > 1)).any()):
        raise ValueError('positive weights and unit probabilities required')
    weights = weights.float()/weights.sum()
    hard, truth = prediction >= .5, label >= .5
    positive, negative = weights[truth].sum(), weights[~truth].sum()
    sensitivity = float(weights[truth & hard].sum()/positive) if positive > 0 else None
    specificity = float(weights[~truth & ~hard].sum()/negative) if negative > 0 else None
    return {'soft_mask_mae': float(((prediction-label).abs()*weights).sum()),
        'soft_mask_mse': float(((prediction-label).square()*weights).sum()),
        'binary_agreement': float(((hard == truth)*weights).sum()),
        'balanced_accuracy': (sensitivity+specificity)/2 if sensitivity is not None and specificity is not None else None,
        'true_positive_rate': sensitivity, 'true_negative_rate': specificity,
        'predicted_mean': float((prediction*weights).sum()),
        'target_mean': float((label*weights).sum()), 'predicted_min': float(prediction.min()),
        'predicted_max': float(prediction.max())}


def history_inputs(capture, past_masks, key_name):
    ids = capture['frame_ids'][0, 0]
    tokens = capture['token_ids'][0, 0]
    prototypes, roles = [], []
    for frame in list(dict.fromkeys(ids.tolist())):
        if frame < 0 or frame >= len(past_masks):
            raise ValueError('history role attempted current/future frame access')
        selected = torch.nonzero(ids == frame).flatten()
        selected = selected[tokens[selected].argsort()]
        if not torch.equal(tokens[selected], torch.arange(1560)):
            raise ValueError('complete within-frame candidate KV required')
        key = capture[key_name].index_select(1, selected).permute(0, 2, 1, 3).float()
        prototypes.extend(key[:, :, start:start+64].mean(2) for start in range(0, 1560, 64))
        roles.append(group_means(past_masks[frame].flatten()))
    k = torch.stack(prototypes, dim=2)
    role = torch.cat(roles).view(1, 1, -1).expand(k.shape[:3])
    return k, role


@torch.inference_mode()
def probe(capture, past_masks):
    """No current mask argument; past prefix is already sliced by the caller."""
    start = capture['current_start']//1560
    if len(past_masks) != start:
        raise ValueError('exact completed prefix required')
    qtokens = capture['query'].shape[1]
    if qtokens % 1560:
        raise ValueError('complete query frames required')
    spatial = group_means(past_masks[-1].flatten().repeat(qtokens//1560))
    predictions = {'past_spatial_persistence': spatial,
                   'past_area_constant': torch.full_like(spatial, float(past_masks[-1].mean()))}
    head_predictions = {}
    for space, qname, kname in [('raw', 'query_unrotated', 'key_unrotated'), ('post_rope', 'query', 'key')]:
        summary = summarize_query_for_pretransfer(capture[qname], 64,
            coordinate_space='unrotated' if space == 'raw' else 'post_rope')
        k, role = history_inputs(capture, past_masks, kname)
        probability = causal_query_identity_probability(summary.query_centroids, k, role)
        predictions[f'{space}_q_only'] = probability.mean(1)[0]
        predictions[f'{space}_q_plus_spatial50'] = (.5*probability+.5*spatial.view(1, 1, -1)).mean(1)[0]
        head_predictions[space] = probability[0]
    return predictions, head_predictions, summary.query_group_sizes[0, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    workspace = Path(args.workspace).resolve()
    teacher_root = workspace/'results/metrics/sam2_oracle_f9d5ef0_v2/state'
    producer = json.loads((teacher_root/'result.json').read_text())
    if producer['status'] != 'pass' or producer['online_method'] or producer['manual_roi_used']:
        raise ValueError('audited automatic offline teacher required')
    masks = torch.load(teacher_root/'teacher_masks.pt', map_location='cpu', weights_only=True)['latent_anchor_masks']
    capture_root = workspace/'results/metrics/matched_trajectory_capture_be00491'
    trajectory = json.loads((capture_root/'trajectory_audit.json').read_text())
    actor = [r for r in trajectory['cases'] if r['method'] == 'rag_dense' and r['prompt'] == 'calibration_state']
    if trajectory['status'] != 'pass' or len(actor) != 1:
        raise ValueError('exact same Dense actor required')
    states = json.loads((workspace/'results/videos/aligned_final_cf5c25f_local/states.json').read_text())['cases']
    dense = next(c for c in states if c['id'] == actor[0]['id'])
    if dense['video_sha256'] != producer['video_sha256'] or file_sha(dense['video']) != producer['video_sha256']:
        raise ValueError('teacher mask/video actor mismatch')
    tasks = json.loads((workspace/'results/metrics/cross_route_plans_eff339a/state/tasks.json').read_text())['tasks']
    tasks = [t for t in tasks if t['actor'] == 'rag_dense']
    if len(tasks) != 8:
        raise ValueError('four layers x two history points required')
    rows = []
    for task in tasks:
        for name, expected in zip(task['capture_files'], task['capture_sha256']):
            path = capture_root/name
            if file_sha(path) != expected:
                raise ValueError('capture checksum changed')
            capture = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            start = capture['current_start']//1560
            # Predictions finish before current labels are indexed.
            predictions, heads, weights = probe(capture, masks[:start].clone())
            qframes = capture['query'].shape[1]//1560
            target = group_means(masks[start:start+qframes].flatten())
            metrics = {k: role_metrics(v, target, weights) for k, v in predictions.items()}
            row = {'layer': capture['layer'], 'start_latent': start, 'denoising_call': capture['denoising_pass'],
                'candidate_frames': capture['frame_ids'][0, 0].unique().tolist(), 'metrics': metrics,
                'per_head_metrics': {s: [role_metrics(v, target, weights) for v in p] for s, p in heads.items()},
                'capture_sha256': expected}
            rows.append(row)
            print(json.dumps({'layer': row['layer'], 'start': start, 'call': row['denoising_call'],
                'mae': {k: v['soft_mask_mae'] for k, v in metrics.items()}}), flush=True)
    result = {'status': 'pass', 'calls': len(rows), 'records': rows,
        'scope': 'oracle_history_assisted_query_role_diagnostic_on_matching_Dense_actor',
        'teacher_mask_sha256': file_sha(teacher_root/'teacher_masks.pt'),
        'current_masks_only_used_after_predictions_for_evaluation': True,
        'past_masks_from_offline_oracle_not_a_causal_runtime_claim': True,
        'CPU_FP32_reduction_diagnostic_not_bitwise_GPU_proxy_reproduction': True,
        'fixed_formulas_no_temperature_or_layer_selection': True,
        'single_state_prompt_single_seed_not_semantic_ground_truth': True,
        'formal_promotion': False}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
