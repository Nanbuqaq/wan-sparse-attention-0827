#!/usr/bin/env python3
"""Same-admission Dense/Final profiles of generic materialization and cache."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.case_identity import build_case_identity
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from adapters.longlive_sparse.methods import method_spec


def build(commit, *, latent_frames=39, prompt_id='calibration_motion', lanes=(0, 1, 2, 3), method='rag_dense'):
    prompts = json.loads((ROOT/'configs/system/profile_calibration_prompts.json').read_text())['candidates']
    prompt = next(prompt for prompt in prompts if prompt['prompt_id'] == prompt_id)
    if latent_frames not in (39, 120, 240) or not lanes or len(set(lanes)) != len(lanes) or any(lane not in range(4) for lane in lanes):
        raise ValueError('invalid development profile shape/lanes')
    if method not in ('rag_dense', 'transfer_vaware_hybrid_history'):
        raise ValueError('only validated Dense/legacy Final admission is allowed')
    params = (json.loads((ROOT/'configs/formal/method_params.json').read_text())['method_params'][method]
              if method != 'rag_dense' else {})
    density = 1.0 if method == 'rag_dense' else .25
    systems = [
        LongLiveSystemConfig(),
        LongLiveSystemConfig(transfer_layout='exact_compact', staging_mode='persistent_separate'),
        LongLiveSystemConfig(transfer_layout='exact_compact', staging_mode='persistent_separate', cpu_pack_policy='archive_runs'),
        LongLiveSystemConfig(transfer_layout='exact_compact', staging_mode='persistent_separate', cpu_pack_policy='archive_runs',
                            gpu_union_cache='per_chunk', gpu_union_cache_budget_mib=4096 if method=='rag_dense' else 768),
    ]
    suites, expected = {}, []
    for lane, system in enumerate(systems):
        if lane not in lanes:
            continue
        case = {**prompt, 'seed': 20260904, 'latent_frames': latent_frames, 'record_per_call': True,
                'longlive_system': system.as_dict()}
        suites[lane] = {'status': 'frozen_same_admission_development_system_validation',
            'experiment_commit': commit, 'formal_prompts_used': False,
            'methods': [method], 'method_params': {method:params}, 'backend': 'grouped_fa2', 'history_density': density,
            'refresh_policy': 'per_chunk', 'rope_policy': 'upstream_zero', 'cases': [case]}
        expected.append({**build_case_identity(commit=commit, method=method, prompt_id=case['prompt_id'],
            prompt=case['prompt'], seed=case['seed'], latent_frames=latent_frames, history_density=density,
            rope_policy='upstream_zero', refresh_policy='per_chunk', backend='grouped_fa2',
            system_identity=system.identity_dict(), method_params=params),
            'method': method, 'backend': 'grouped_fa2', 'routing_stage': method_spec(method).routing_stage,
            'latent_frames': latent_frames, 'lane': lane})
    assert len({item['case_key_sha256'] for item in expected}) == len(lanes)
    return suites, {'cases': expected}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--latent-frames', type=int, default=39)
    parser.add_argument('--prompt-id', default='calibration_motion')
    parser.add_argument('--lanes', default='0,1,2,3')
    parser.add_argument('--method', default='rag_dense', choices=('rag_dense', 'transfer_vaware_hybrid_history'))
    args = parser.parse_args()
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    suites, expected = build(commit, latent_frames=args.latent_frames, prompt_id=args.prompt_id,
                             lanes=tuple(int(lane) for lane in args.lanes.split(',')), method=args.method)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for lane, suite in suites.items():
        (root/f'lane{lane}.json').write_text(json.dumps(suite, indent=2)+'\n')
    (root/'expected.json').write_text(json.dumps(expected, indent=2)+'\n')


if __name__ == '__main__':
    main()
