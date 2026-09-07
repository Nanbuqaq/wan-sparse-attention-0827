#!/usr/bin/env python3
"""Freeze two development lanes for same-route metadata-control ablation."""
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


def build(commit, latent_frames):
    if latent_frames not in (39, 120, 240):
        raise ValueError('only declared development lengths are supported')
    prompts = json.loads((ROOT/'configs/system/profile_calibration_prompts.json').read_text())['candidates']
    params = json.loads((ROOT/'configs/formal/method_params.json').read_text())['method_params']
    suites, expected = {}, []
    for lane, kind in enumerate(('motion', 'state')):
        prompt = next(p for p in prompts if p['prompt_id'] == 'calibration_' + kind)
        methods = ['rag_dense', 'transfer_vaware_hybrid_history']
        modes = ['recompute', 'validated_reuse']
        if lane:
            modes.reverse(); methods.reverse()
        cases = []
        for method in methods:
            for mode in modes:
                system = LongLiveSystemConfig(profile_mode='summary', transfer_layout='exact_compact',
                    staging_mode='persistent_separate', cpu_pack_policy='archive_runs',
                    gpu_union_cache='per_chunk', gpu_union_cache_budget_mib=4096 if method == 'rag_dense' else 768,
                    archive_offload='pooled_pageable', host_pinned_budget_mib=128, route_metadata_mode=mode)
                case = dict(prompt, seed=20260904, latent_frames=latent_frames, record_per_call=True,
                            only_method=method, longlive_system=system.as_dict())
                cases.append(case)
                expected.append(dict(build_case_identity(commit=commit, method=method, prompt_id=prompt['prompt_id'],
                    prompt=prompt['prompt'], seed=case['seed'], latent_frames=latent_frames,
                    history_density=1. if method == 'rag_dense' else .25, rope_policy='upstream_zero',
                    refresh_policy='per_chunk', backend='grouped_fa2', system_identity=system.identity_dict(),
                    method_params=params.get(method, {})), lane=lane, method=method, metadata_mode=mode))
        suites[lane] = dict(status='frozen_development_metadata_ablation', experiment_commit=commit,
            methods=methods, method_params={m: params.get(m, {}) for m in methods}, backend='grouped_fa2',
            history_density=.25, method_history_densities={'rag_dense': 1.}, refresh_policy='per_chunk',
            rope_policy='upstream_zero', cases=cases, formal_prompts_used=False,
            independent_timing_repeats=False, scope='same_loaded_model_control_then_opposite_order_on_other_prompt')
    assert len({c['case_key_sha256'] for c in expected}) == 8
    return suites, expected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--latent-frames', type=int, required=True)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    suites, expected = build(commit, args.latent_frames)
    if args.validate_only:
        print(json.dumps({'status': 'validated', 'cases': len(expected), 'latent_frames': args.latent_frames}))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for lane, suite in suites.items():
        raw = json.dumps(suite, indent=2) + '\n'
        (args.output_dir/f'lane{lane}.json').write_text(raw)
        manifests[str(lane)] = hashlib.sha256(raw.encode()).hexdigest()
    (args.output_dir/'expected.json').write_text(json.dumps({'cases': expected, 'suite_sha256': manifests}, indent=2)+'\n')
    print(json.dumps({'status': 'frozen', 'output': str(args.output_dir), 'source': commit, 'suite_sha256': manifests}))


if __name__ == '__main__':
    main()
