#!/usr/bin/env python3
"""One new Dense hash control and two isolated oracle-addressing videos."""
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


def build(commit, teacher_path):
    teacher_path = Path(teacher_path).resolve()
    producer = json.loads(teacher_path.with_name('result.json').read_text())
    if producer['status'] != 'pass' or producer['manual_roi_used'] or producer['latent_frames'] != 39:
        raise ValueError('matching automatic153-frame offline teacher required')
    prompts = json.loads((ROOT/'configs/system/profile_calibration_prompts.json').read_text())['candidates']
    prompt = next(p for p in prompts if p['prompt_id'] == 'calibration_state')
    common = {**prompt, 'seed': 20260904, 'latent_frames': 39, 'record_per_call': True, 'complete_capture': False}
    base = dict(transfer_layout='exact_compact', staging_mode='persistent_separate', cpu_pack_policy='archive_runs',
        gpu_union_cache='per_chunk', gpu_union_cache_budget_mib=4096, archive_offload='pooled_pageable', host_pinned_budget_mib=128)
    cases = [{**common, 'only_method': 'rag_dense', 'backend': 'grouped_fa2', 'method_params': {},
              'longlive_system': LongLiveSystemConfig(**base).as_dict()}]
    for timeline in ('source_compatible_addressing', 'aligned_latent_anchors'):
        params = {'oracle_timeline': timeline, 'oracle_mask_sha256': hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
            'oracle_reference_video_sha256': producer['video_sha256'], 'target_average': .25, 'age_decay_floor': .05}
        cases.append({**common, 'only_method': 'tethermem_oracle_mask_teacher', 'method_params': params,
            'backend': 'split_role_sdpa_reference', 'required_reference_video_sha256': producer['video_sha256'],
            'longlive_system': LongLiveSystemConfig(**base, execution_dataflow='biased_sdpa_reference').as_dict()})
    suite = {'status': 'frozen_offline_Tether_mechanism_no_online_Pareto', 'experiment_commit': commit,
        'methods': ['rag_dense', 'tethermem_oracle_mask_teacher'], 'method_params': {}, 'cases': cases,
        'history_density': 1., 'backend': 'grouped_fa2', 'rope_policy': 'upstream_zero', 'refresh_policy': 'per_chunk',
        'formal_prompts_used': False, 'causal_online': False, 'reference_video_generation_counted_as_separate_case': True,
        'SAM2_producer_costs': producer, 'SAM2_result_sha256': hashlib.sha256(teacher_path.with_name('result.json').read_bytes()).hexdigest()}
    expected = []
    for case in cases:
        identity = build_case_identity(commit=commit, method=case['only_method'], prompt_id=case['prompt_id'],
            prompt=case['prompt'], seed=case['seed'], latent_frames=39, history_density=1., backend=case['backend'],
            rope_policy='upstream_zero', refresh_policy='per_chunk', method_params=case['method_params'],
            system_identity=case['longlive_system'])
        expected.append({**identity, 'method': case['only_method'], 'latent_frames': 39, 'seed': case['seed'],
                         'prompt_id': case['prompt_id'], 'online_Pareto_eligible': False})
    return suite, {'scope': 'offline_oracle_and_matched_Dense_control', 'cases': expected}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--teacher', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    source = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    suite, expected = build(source, args.teacher)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out/'suite.json').write_text(json.dumps(suite, indent=2)+'\n')
    (out/'expected.json').write_text(json.dumps(expected, indent=2)+'\n')
    print(json.dumps({'status': 'pass', 'cases': 3, 'source': source}))


if __name__ == '__main__':
    main()
