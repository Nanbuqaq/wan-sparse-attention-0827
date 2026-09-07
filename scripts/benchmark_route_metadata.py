#!/usr/bin/env python3
"""CPU-only route-control replay with exact cache identities and full window cost."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.rope import build_sparse_positions
from adapters.longlive_sparse.route_plan import HistoryRoutePlan
from adapters.longlive_sparse.route_metadata import RouteIdentityCache


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    audit = json.loads((args.workspace/'results/metrics/matched_trajectory_capture_be00491/trajectory_audit.json').read_text())
    report = {'status': 'running', 'scope': 'CPU_control_replay_not_GPU_or_video_speedup',
        'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('scripts/benchmark_route_metadata.py', 'adapters/longlive_sparse/route_metadata.py',
             'adapters/longlive_sparse/route_plan.py')}, 'warmup': 5, 'repeats': args.repeats, 'records': []}
    for case in audit['cases']:
        if case['method'] not in ('rag_dense', 'transfer_vaware_hybrid_history'):
            continue
        path = Path(case['capture_dir'])/'layer09_start00046800_pass00.pt'
        capture = torch.load(path, map_location='cpu', weights_only=True)
        plan = HistoryRoutePlan.from_state_dict(capture['route_plan'])
        candidates = tuple(int(x) for x in torch.unique(capture['frame_ids'], sorted=True))
        options = dict(current_frame_id=30, spatial_width=52, rope_policy='upstream_zero',
                       max_relative_age=1024, candidate_frame_ids=candidates)
        del capture
        def legacy():
            positions = build_sparse_positions(frame_ids=plan.union_frame_ids.clamp_min(0),
                token_ids=plan.union_token_ids.clamp_min(0), current_frame_id=30,
                spatial_width=52, rope_policy='upstream_zero', max_relative_age=1024,
                candidate_frame_ids=torch.tensor(candidates))
            coordinates = torch.stack((plan.union_frame_ids.long(), plan.union_token_ids.long()), dim=-1)
            return (plan.digest(), tensor_sha256(coordinates), tensor_sha256(positions))
        cache = RouteIdentityCache()
        expected = legacy()
        samples = {'recompute': [], 'validated_reuse': []}
        cold, warm = [], []
        for repeat in range(5 + args.repeats):
            for variant in (('recompute', 'validated_reuse') if repeat % 2 == 0 else ('validated_reuse', 'recompute')):
                cache.clear()
                begin = time.perf_counter()
                calls = []
                for invocation in range(5):
                    started = time.perf_counter()
                    if variant == 'recompute':
                        observed = legacy()
                    else:
                        identity = cache.prepare(plan, **options)
                        observed = (identity.route_sha256, identity.coordinate_sha256, identity.position_sha256)
                    # Additional independent consumers exist in stats/backend.
                    assert plan.digest() == expected[0]
                    assert plan.digest() == expected[0]
                    calls.append(time.perf_counter() - started)
                    assert observed == expected
                elapsed = time.perf_counter() - begin
                if repeat >= 5:
                    samples[variant].append(elapsed)
                    if variant == 'validated_reuse':
                        cold.append(calls[0]); warm.extend(calls[1:])
        cache.prepare(plan, **options)
        row = {'method': case['method'], 'prompt': case['prompt'], 'capture': str(path),
            'route_sha256': expected[0], 'status': 'pass', 'five_call_samples_s': samples,
            'median_s': {k: statistics.median(v) for k, v in samples.items()},
            'p95_s': {k: float(np.percentile(v, 95)) for k, v in samples.items()},
            'reuse_cold_call_median_s': statistics.median(cold), 'reuse_warm_call_median_s': statistics.median(warm),
            'extra_CPU_bytes': cache.retained_CPU_bytes, 'identity_exact': True}
        row['five_call_speedup'] = row['median_s']['recompute']/row['median_s']['validated_reuse']
        report['records'].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'five_call_samples_s'}), flush=True)
        (args.output/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
        cache.clear()
    report['status'] = 'pass'
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
