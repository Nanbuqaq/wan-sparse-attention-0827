#!/usr/bin/env python3
"""Matched frozen-mode audit for runtime smoke and formal video batches."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_system_video_comparison import load_case, compare


def audit(root):
    expected_path = root/'control/expected.json'
    expected = json.loads(expected_path.read_text())
    paths = list(root.glob('lane*/*/case_state.json'))
    states = {}
    for path in paths:
        state = json.loads(path.read_text())
        if state['id'] in states:
            raise ValueError('duplicate terminal case')
        states[state['id']] = (state, path.parent)
    wanted = {c['id']: c for c in expected['cases']}
    if states.keys() != wanted.keys():
        raise ValueError('unexpected or missing case states')
    groups = defaultdict(dict)
    for identity, (state, path) in states.items():
        frozen = wanted[identity]
        if state['case_key'] != frozen['case_key']:
            raise ValueError('frozen/runtime identity mismatch')
        groups[(frozen['prompt_id'], frozen['seed'], frozen['latent_frames'])][frozen['formal_config_id']] = (state, path)
    rows = []
    for (prompt, seed, length), configs in sorted(groups.items()):
        if set(configs) != {'rag_dense', 'legacy_final', 'legacy_final_system'}:
            raise ValueError('expected exactly three frozen configurations')
        if any(s['status'] != 'pass' for s, p in configs.values()):
            rows.append({'prompt': prompt, 'seed': seed, 'latent_frames': length, 'status': 'fail',
                         'failure_scope': 'technical_case_failure',
                         'cases': {k: {'id': s['id'], 'status': s['status']} for k, (s, p) in configs.items()}})
            continue
        if len({p.parent for s, p in configs.values()}) != 1:
            raise ValueError('group did not run in the same GPU lane')
        if len({s['initial_noise_sha256'] for s, p in configs.values()}) != 1:
            raise ValueError('unmatched noise across frozen configurations')
        loaded = {k: load_case(p) for k, (s, p) in configs.items()}
        pair = compare(configs['legacy_final'][1], configs['legacy_final_system'][1])
        same_raw = configs['legacy_final'][0].get('raw_video_rgb_sha256') == configs['legacy_final_system'][0].get('raw_video_rgb_sha256')
        same_raw = same_raw and configs['legacy_final'][0].get('raw_video_rgb_sha256') is not None
        passed = pair['status'] == 'pass' and pair['bitwise_equal_latents'] and same_raw
        dense_time = configs['rag_dense'][0]['end_to_end_s']
        summary = {}
        for k, (state, path) in configs.items():
            stats = loaded[k][2]
            summary[k] = {'case_id': state['id'], 'end_to_end_s': state['end_to_end_s'],
                'speedup_vs_same_lane_dense': dense_time/state['end_to_end_s'],
                'model_load_s_total': state['model_load_s_total'], 'wall_breakdown': state.get('wall_breakdown'),
                'peak_allocated_gb': state['peak_allocated_gb'], 'history_pair_density': state['history_pair_density'],
                'history_transfer_density': state['history_transfer_density'],
                'global_executed_density': state['global_executed_density'],
                'transferred_bytes': state['transferred_bytes'], 'archive_storage': stats.get('archive_storage'),
                'history_union_cache': stats.get('history_union_cache'),
                'all_phases_charged': True, 'state_sha256': hashlib.sha256((path/'case_state.json').read_bytes()).hexdigest()}
        rows.append({'prompt': prompt, 'seed': seed, 'latent_frames': length,
            'status': 'pass' if passed else 'fail', 'same_noise': True,
            'same_final_routes': pair['same_ordered_routes'], 'same_final_latents': pair['bitwise_equal_latents'],
            'same_final_preencode_pixels': same_raw, 'system_speedup': pair['end_to_end_speedup'],
            'system_complete_reduction': pair['end_to_end_reduction'], 'cases': summary})
    return {'status': 'pass' if all(r['status'] == 'pass' for r in rows) else 'fail',
        'expected_cases': len(wanted), 'technical_pass': sum(s['status'] == 'pass' for s, p in states.values()),
        'missing': 0, 'groups': rows, 'expected_sha256': hashlib.sha256(expected_path.read_bytes()).hexdigest(),
        'scope': 'same_gpu_frozen_modes_full_latent_route_and_raw_pixel_equivalence',
        'absolute_quality_review_separate': True, 'formal_results_select_no_new_parameters': True,
        'no_new_admission_or_complete_codesign_algorithm_claim': True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    result = audit(args.root.resolve())
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'groups'}))
    if result['status'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
