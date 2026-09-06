#!/usr/bin/env python3
"""Freeze only evidence-backed online configurations before formal results."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from adapters.longlive_sparse.system_formal import validate_system_method_freeze


def locked(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get('status') != 'pass':
        raise ValueError(f'passing artifact audit required: {path}')
    return data, {'artifact': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', required=True)
    p.add_argument('--batched-audit', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    workspace = Path(args.workspace).resolve()
    metrics = workspace/'results/metrics'
    bootstrap, calibration = locked(metrics/'bootstrap477_quality/decision.json')
    raw, quality = locked(metrics/'preencode_diagnostic_59540bd_audit.json')
    system, system_audit = locked(metrics/'shared_compiler_repeats_audit_20260906.json')
    batched, batched_lock = locked(args.batched_audit)
    profiles = []
    for method in ('rag_dense', 'transfer_vaware_hybrid_history'):
        result, lock = locked(metrics/f'optimized_length_profiles_0ea5d25/{method}/summary.json')
        if {r['latent_frames'] for r in result['records']} != {39, 120, 240}:
            raise ValueError('all optimized39/120/240 profiles required')
        profiles.append(lock)
    profile_digest = hashlib.sha256(json.dumps(profiles, sort_keys=True).encode()).hexdigest()
    if bootstrap['formal_promotion'] or not all(g['all_latents_equal'] and g['ordered_routes_equal'] for g in raw['groups']):
        raise ValueError('unexpected calibration or raw equivalence decision')
    if system['technical_successes'] != 8 or system['missing'] != 0:
        raise ValueError('repeated system gate incomplete')
    if len(system['groups']) != 2 or not all(
        g['summary']['end_to_end_s']['reduction'] > 0 and
        all(r['same_ordered_routes'] and r['same_latent_video_bytes'] for r in g['records'])
        for g in system['groups']
    ):
        raise ValueError('repeated compiler gate must preserve routes/latents and improve both categories')
    use_batched = bool(batched['eligible_for_formal_system_config'])
    backend = 'batched_fa2' if use_batched else 'grouped_fa2'
    flow = 'qout_batched_fa2' if use_batched else 'qout_grouped_fa2'
    common = dict(archive_offload='pooled_pageable', host_pinned_budget_mib=128,
                  staging_mode='persistent_separate', cpu_threads=2)
    optimized = LongLiveSystemConfig(**common, transfer_layout='exact_compact', cpu_pack_policy='archive_runs',
        gpu_union_cache='per_chunk', gpu_union_cache_budget_mib=4096, execution_dataflow=flow)
    legacy = LongLiveSystemConfig(**common)
    configs = [
        {'config_id': 'rag_dense', 'method': 'rag_dense', 'online': True, 'backend': backend,
         'history_density': 1., 'longlive_system': optimized.as_dict()},
        {'config_id': 'legacy_final', 'method': 'transfer_vaware_hybrid_history', 'online': True,
         'backend': 'grouped_fa2', 'history_density': .25, 'longlive_system': legacy.as_dict()},
        {'config_id': 'legacy_final_system', 'method': 'transfer_vaware_hybrid_history', 'online': True,
         'backend': backend, 'history_density': .25, 'longlive_system': optimized.as_dict()},
    ]
    for config in configs:
        config.update(rope_policy='upstream_zero', refresh_policy='per_chunk')
    result = {'artifact_id': 'longlive_system_method_freeze_v1', 'status': 'frozen_after_system_calibration',
        'formal_results_used': False, 'configs': configs, 'calibration_audit': calibration,
        'quality_gate': quality, 'profile_audit': {'sha256': profile_digest, 'sources': profiles},
        'system_equivalence_audit': system_audit, 'batched_backend_audit': batched_lock,
        'batched_backend_promoted': use_batched,
        'admission_params': 'unchanged frozen legacy70/15/15 V-aware Final; no new admission promoted',
        'negative_decisions': {'cost_aware_admission': 'no validated <=15% MAPE model',
            'new_utility_and_alignment': 'did not pass two-category/independent long quality gates',
            'hierarchical_raw_cache': 'mixed complete-time result despite lower KV traffic',
            'causal_roles': 'motion mask feasibility and two-category role-prediction gates not passed',
            'oracle_tether': 'completed offline teacher, never eligible for online Pareto',
            'KVOut_video': 'no optimized runtime/backend passed the real-video critical-path promotion gate'},
        'quality_protocol': 'canonical_raw_VAE_u8_pixels; MP4 previews used for integrity only',
        'fair_comparisons': {'algorithm': ['rag_dense', 'legacy_final_system'],
                             'system': ['legacy_final', 'legacy_final_system']},
        'common_fixes': ['D2H prototype readiness', 'bounded pageable archive staging', 'two CPU threads'],
        'total_CPU_archive_is_not_bounded': True,
        'no_new_admission_or_complete_codesign_algorithm_claim': True}
    errors = validate_system_method_freeze(result)
    if errors:
        raise ValueError(errors)
    out = Path(args.output)
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': 'pass', 'configs': [c['config_id'] for c in configs], 'backend': backend,
                      'freeze_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
