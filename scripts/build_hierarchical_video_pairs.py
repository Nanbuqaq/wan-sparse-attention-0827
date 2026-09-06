#!/usr/bin/env python3
"""Eight matched 477 cases: Dense/Final x motion/state x union/hierarchy."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.case_identity import build_case_identity
from adapters.longlive_sparse.system_config import LongLiveSystemConfig


def build(commit, latent_frames=120, *, raw_video_capture=False, lane_filter=None):
    if latent_frames not in (39, 120):
        raise ValueError('isolated development lengths only')
    path = ROOT/'configs/system/profile_calibration_prompts.json'
    prompts = {p['prompt_id']: p for p in json.loads(path.read_text())['candidates']}
    final = json.loads((ROOT/'configs/formal/method_params.json').read_text())['method_params']['transfer_vaware_hybrid_history']
    suites, expected = {}, []
    lanes = [('motion', 'rag_dense'), ('motion', 'transfer_vaware_hybrid_history'),
             ('state', 'rag_dense'), ('state', 'transfer_vaware_hybrid_history')]
    for lane, (kind, method) in enumerate(lanes):
        if lane_filter is not None and lane not in lane_filter:
            continue
        density = 1. if method == 'rag_dense' else .25
        params = {} if method == 'rag_dense' else final
        cases = []
        # Counterbalance systems across categories for each method.
        for mode in (('per_chunk', 'hierarchical') if kind == 'motion' else ('hierarchical', 'per_chunk')):
            system = LongLiveSystemConfig(transfer_layout='exact_compact', staging_mode='persistent_separate',
                cpu_pack_policy='archive_runs', gpu_union_cache=mode, gpu_union_cache_budget_mib=4096,
                raw_cache_budget_mib=1024 if mode == 'hierarchical' else 0,
                archive_offload='pooled_pageable', host_pinned_budget_mib=128,
                profile_mode='trace' if raw_video_capture else 'off')
            case = {**prompts[f'calibration_{kind}'], 'seed': 20260904, 'latent_frames': latent_frames,
                    'record_per_call': True, 'complete_capture': False, 'longlive_system': system.as_dict(),
                    'raw_video_capture': raw_video_capture}
            cases.append(case)
            identity = build_case_identity(commit=commit, method=method, prompt_id=case['prompt_id'], prompt=case['prompt'],
                seed=case['seed'], latent_frames=latent_frames, history_density=density, backend='grouped_fa2',
                rope_policy='upstream_zero', refresh_policy='per_chunk', system_identity=system.identity_dict(), method_params=params)
            expected.append({**identity, 'method': method, 'lane': lane, 'cache_mode': mode,
                             'latent_frames': latent_frames, 'prompt_id': case['prompt_id'], 'seed': case['seed']})
        suites[lane] = {'status': 'frozen_hierarchical_system_matched_development', 'experiment_commit': commit,
            'formal_prompts_used': False, 'methods': [method], 'method_params': {method: params}, 'cases': cases,
            'history_density': density, 'backend': 'grouped_fa2', 'rope_policy': 'upstream_zero', 'refresh_policy': 'per_chunk',
            'same_gpu_same_loaded_model_pair': True, 'source_prompts_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'same_total_configured_cache_budget_mib': 4096, 'actual_allocated_cache_reported_separately': True,
            'promotion_gate': 'identical ordered routes, latent/video bytes; complete costs including index H2D and backing storage'}
    return suites, {'scope': 'development_system_equivalence_not_new_quality_samples', 'cases': expected}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', required=True)
    p.add_argument('--latent-frames', type=int, default=120)
    p.add_argument('--raw-video-capture', action='store_true')
    p.add_argument('--lanes', default='0,1,2,3')
    args = p.parse_args()
    source = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    lanes = tuple(int(x) for x in args.lanes.split(','))
    if not lanes or len(set(lanes)) != len(lanes) or not set(lanes) <= set(range(4)):
        raise ValueError('invalid lane subset')
    suites, expected = build(source, args.latent_frames, raw_video_capture=args.raw_video_capture, lane_filter=lanes)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    for lane, suite in suites.items():
        (out/f'lane{lane}.json').write_text(json.dumps(suite, indent=2)+'\n')
    (out/'expected.json').write_text(json.dumps(expected, indent=2)+'\n')
    print(json.dumps({'status': 'pass', 'cases': len(expected['cases']), 'source': source}))


if __name__ == '__main__':
    main()
