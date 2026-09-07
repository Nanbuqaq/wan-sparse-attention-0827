#!/usr/bin/env python3
"""Freeze the first group-relation video screen; not a formal holdout stage."""
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
from adapters.longlive_sparse.config import SparseHistoryConfig


def build(commit, latent_frames):
    if latent_frames not in (39, 120, 240):
        raise ValueError('unsupported development length')
    prompts = json.loads((ROOT/'configs/system/profile_calibration_prompts.json').read_text())['candidates']
    final_params = json.loads((ROOT/'configs/formal/method_params.json').read_text())['method_params']['transfer_vaware_hybrid_history']
    suites, expected = {}, []
    for lane, kind in enumerate(('motion', 'state')):
        prompt = next(p for p in prompts if p['prompt_id'] == 'calibration_' + kind)
        variants = [('rag_dense', 'resident_grouped_fa2', {}),
                    ('transfer_vaware_hybrid_history', 'resident_grouped_fa2', final_params)]
        for admission, backend in (('shared', 'resident_grouped_fa2'), ('per_group', 'resident_grouped_fa2'), ('per_group', 'grouped_fa2')):
            variants.append(('group_relation_history', backend, dict(information_grouping='spatial_quadrants',
                            relation_admission=admission, group_start_layer=8)))
        if lane:
            variants.reverse()
        methods = list(dict.fromkeys(m for m, _, _ in variants))
        cases = []
        for method, backend, params in variants:
            system = LongLiveSystemConfig(profile_mode='summary', transfer_layout='exact_compact',
                staging_mode='persistent_separate', cpu_pack_policy='archive_runs', gpu_union_cache='per_chunk',
                gpu_union_cache_budget_mib=4096, archive_offload='pooled_pageable', host_pinned_budget_mib=128,
                route_metadata_mode='validated_reuse', execution_dataflow='qout_resident_grouped_fa2'
                    if backend == 'resident_grouped_fa2' else 'qout_grouped_fa2')
            density = 1. if method == 'rag_dense' else .25
            SparseHistoryConfig(method=method, backend=backend, method_params=params, history_density=density,
                                rope_policy='upstream_zero', refresh_policy='per_chunk')
            case = dict(prompt, seed=20260904, latent_frames=latent_frames, record_per_call=True,
                only_method=method, backend=backend, method_params=params, history_density=density,
                longlive_system=system.as_dict())
            cases.append(case)
            expected.append(dict(build_case_identity(commit=commit, method=method, prompt_id=prompt['prompt_id'],
                prompt=prompt['prompt'], seed=case['seed'], latent_frames=latent_frames, history_density=density,
                rope_policy='upstream_zero', refresh_policy='per_chunk', backend=backend,
                system_identity=system.identity_dict(), method_params=params), lane=lane, method=method,
                backend=backend, relation_admission=params.get('relation_admission')))
        suites[lane] = dict(status='frozen_group_relation_development_screen', experiment_commit=commit,
            methods=methods, cases=cases, backend='resident_grouped_fa2', history_density=.25,
            refresh_policy='per_chunk', rope_policy='upstream_zero', formal_prompts_used=False,
            layer_rule='legacy_Final_layers0_7; declared_group_relation_layers8_29',
            evidence_limit='equal_history_pairs_not_equal_transfer_union_or_actual_GPU_allocation',
            GPU_cache_budget_equal_across_methods=True, independent_timing_repeats=False)
    assert len({c['case_key_sha256'] for c in expected}) == 10
    return suites, expected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--latent-frames', type=int, default=39)
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    suites, expected = build(commit, args.latent_frames)
    if args.validate_only:
        print(json.dumps({'status': 'validated', 'cases': len(expected)}))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shas = {}
    for lane, suite in suites.items():
        raw = json.dumps(suite, indent=2)+'\n'
        (args.output_dir/f'lane{lane}.json').write_text(raw)
        shas[str(lane)] = hashlib.sha256(raw.encode()).hexdigest()
    (args.output_dir/'expected.json').write_text(json.dumps({'cases': expected, 'suite_sha256': shas}, indent=2)+'\n')
    print(json.dumps({'status': 'frozen', 'commit': commit, 'suite_sha256': shas, 'cases': len(expected)}))


if __name__ == '__main__':
    main()
