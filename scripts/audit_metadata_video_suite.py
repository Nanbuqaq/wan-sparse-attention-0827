#!/usr/bin/env python3
"""Audit full latents, RGB, ordered routes, bytes and time for metadata ablations."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_system_video_comparison import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--expected', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    expected = json.loads(args.expected.read_text())['cases']
    cases = [json.loads(path.read_text()) for path in args.root.rglob('case_state.json')]
    if {c['case_key_sha256'] for c in cases} != {c['case_key_sha256'] for c in expected}:
        raise RuntimeError('expected/observed case identities do not match')
    groups = {}
    for c in cases:
        group = (c['method'], c['prompt_id'], c['seed'], c['latent_frames'])
        groups.setdefault(group, {})[c['system']['route_metadata_mode']] = c
    records = []
    for group, pair in sorted(groups.items()):
        if set(pair) != {'recompute', 'validated_reuse'}:
            raise RuntimeError('incomplete metadata pair')
        baseline, candidate = pair['recompute'], pair['validated_reuse']
        control_system, new_system = dict(baseline['system']), dict(candidate['system'])
        control_system.pop('route_metadata_mode'); new_system.pop('route_metadata_mode')
        assert control_system == new_system
        result = compare(Path(baseline['video']).parent, Path(candidate['video']).parent)
        result.update(method=group[0], prompt_id=group[1], seed=group[2], latent_frames=group[3],
            initial_noise_exact=baseline['initial_noise_sha256'] == candidate['initial_noise_sha256'],
            raw_RGB_exact=baseline['raw_video_rgb_sha256'] == candidate['raw_video_rgb_sha256'],
            H2D_bytes_exact=baseline['transferred_bytes'] == candidate['transferred_bytes'],
            inference_speedup=baseline['wall_breakdown']['inference_s']/candidate['wall_breakdown']['inference_s'])
        result['technical_equivalence_pass'] = all(result[k] for k in
            ('same_ordered_routes', 'bitwise_equal_latents', 'initial_noise_exact', 'raw_RGB_exact', 'H2D_bytes_exact'))
        result['performance_status'] = 'positive_single_pair' if result['end_to_end_speedup'] > 1 else 'negative_single_pair'
        records.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)
    report = dict(status='pass' if all(r['technical_equivalence_pass'] for r in records) else 'fail',
        missing=0, cases=len(cases), pairs=records, independent_repeated_timings=False,
        complete_video_timings_not_extrapolated_from_CPU_replay=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
