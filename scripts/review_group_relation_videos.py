#!/usr/bin/env python3
"""Whole-video group screen audit and descriptive boards; no automatic quality winner."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_system_video_comparison import load_case
from scripts.build_video_review_storyboards import analyze
from adapters.longlive_sparse.offline_eval import output_error_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--expected', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    expected = json.loads(args.expected.read_text())['cases']
    states = [json.loads(path.read_text()) for path in args.root.rglob('case_state.json')]
    assert {r['case_key_sha256'] for r in states} == {r['case_key_sha256'] for r in expected}
    groups = {}
    for state in states:
        params = state['case_key'].get('method_params', {})
        variant = state['method'] if state['method'] != 'group_relation_history' else 'group_' + params['relation_admission']
        if state['backend'] == 'grouped_fa2':
            variant += '_old_executor'
        groups.setdefault(state['prompt_id'], {})[variant] = load_case(Path(state['video']).parent)
    results = []
    for prompt, cases in sorted(groups.items()):
        reference = cases['rag_dense']
        new, old = cases['group_per_group'], cases['group_per_group_old_executor']
        for key in ('method', 'prompt', 'seed', 'latent_frames', 'history_density', 'method_params'):
            assert new[3].get(key) == old[3].get(key), key
        executor_equal = bool(torch.equal(new[4], old[4]) and new[5] == old[5]
                              and new[1]['raw_video_rgb_sha256'] == old[1]['raw_video_rgb_sha256'])
        if not executor_equal:
            raise RuntimeError('same logical route changed under the new executor')
        records = []
        for variant, (case, state, stats, config, latent, ordered) in cases.items():
            error = output_error_metrics(reference[4], latent)
            last = output_error_metrics(reference[4][:, -9:], latent[:, -9:])
            row = dict(variant=variant, case=str(case), video=state['video'], status=state['status'],
                source_commit=state['execution_commit'], case_identity=state['case_key_sha256'],
                end_to_end_s=state['end_to_end_s'], inference_s=state['wall_breakdown']['inference_s'],
                peak_allocated_gb=state['peak_allocated_gb'], history_pair_density=state['history_pair_density'],
                H2D_over_cumulative_candidate_bytes=state['history_transfer_density'],
                H2D_bytes=state['transferred_bytes'], raw_RGB_sha256=state['raw_video_rgb_sha256'],
                latent_error_not_absolute_quality=error, last9_latents_error_not_absolute_quality=last,
                ordered_route_sha256=hashlib.sha256(json.dumps(ordered).encode()).hexdigest(),
                recorded_backend_metadata_H2D_bytes=sum(r.get('backend_metadata_H2D_bytes') or 0 for r in stats['call_records']),
                backend_metadata_accounting_available=state['backend'] == 'resident_grouped_fa2')
            if not variant.endswith('old_executor'):
                row['diagnostics'] = analyze(dict(state, id=prompt+'__'+variant), args.output, samples_per_quarter=16)
            records.append(row)
        results.append(dict(prompt=prompt, cases=records, same_route_executor_equivalence=executor_equal,
            same_route_executor_speedup=old[1]['end_to_end_s']/new[1]['end_to_end_s']))
        print(json.dumps({'prompt': prompt, 'same_route_executor_speedup': results[-1]['same_route_executor_speedup'],
            'cases': [{'variant': r['variant'], 'time': r['end_to_end_s'],
                       'relative_l2': r['latent_error_not_absolute_quality']['relative_l2']} for r in records]}), flush=True)
    result = dict(status='pass', missing=0, cases=len(states), groups=results,
        absolute_quality_winner=None, visual_review='descriptive_nonblind_AI_review_pending',
        comparison='same_backbone_prompt_seed; equal_history_pairs_not_equal_H2D_or_memory',
        existing_Dense_and_Final_given_same_resident_executor=True)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
