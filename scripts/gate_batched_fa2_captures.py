#!/usr/bin/env python3
"""Real captured Dense/Final inputs: complete grouped vs route-proven batched FA2."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.backends import execute_plan
from adapters.longlive_sparse.offline_eval import dense_history_attention, output_error_metrics
from adapters.longlive_sparse.route_plan import HistoryRoutePlan, map_union_coordinates


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', required=True)
    p.add_argument('--kind', choices=('motion', 'state'), required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('real CUDA required')
    root = Path(args.workspace).resolve()/'results/metrics/matched_trajectory_capture_be00491'
    audit = json.loads((root/'trajectory_audit.json').read_text())
    rows = []
    for method in ('rag_dense', 'transfer_vaware_hybrid_history'):
        case = next(c for c in audit['cases'] if c['method'] == method and c['prompt'] == f'calibration_{args.kind}')
        path = Path(case['capture_dir'])/'layer09_start00046800_pass00.pt'
        capture = torch.load(path, map_location='cpu', weights_only=True)
        plan = HistoryRoutePlan.from_state_dict(capture['route_plan'])
        indices = map_union_coordinates(plan, capture['frame_ids'], capture['token_ids']).cuda()
        q, ek, ev = [capture[n].cuda() for n in ('query', 'exact_key', 'exact_value')]
        selected = []
        for name in ('key', 'value'):
            value = capture[name].cuda().permute(0, 2, 1, 3)
            selected.append(value.gather(2, indices[..., None].expand(-1, -1, -1, value.shape[-1])).permute(0, 2, 1, 3))
        hk, hv = selected
        reference = dense_history_attention(q, torch.cat((ek, hk), 1), torch.cat((ev, hv), 1))
        initial = {}
        samples = {'grouped_fa2': [], 'batched_fa2': []}
        for repeat in range(35):
            for backend in (('grouped_fa2', 'batched_fa2') if repeat%2 == 0 else ('batched_fa2', 'grouped_fa2')):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = execute_plan(backend, q, ek, ev, hk, hv, plan)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                if backend not in initial:
                    initial[backend] = result.output.clone()
                if repeat >= 5:
                    samples[backend].append(elapsed)
        error = output_error_metrics(reference, initial['batched_fa2'])
        passed = error['max_abs'] <= .02 and error['relative_l2'] <= .01 and error['one_minus_cosine'] <= .001
        row = {'method': method, 'prompt': args.kind, 'status': 'pass' if passed else 'fail',
            'bitwise_grouped_output': torch.equal(initial['grouped_fa2'], initial['batched_fa2']),
            'bf16_vs_fp32': error, 'route_sha': plan.digest(), 'samples_s': samples,
            'median_s': {k: statistics.median(v) for k, v in samples.items()},
            'complete_backend_reduction': 1-statistics.median(samples['batched_fa2'])/statistics.median(samples['grouped_fa2']),
            'scope': 'complete_eligibility_concat_kernel_no_transfer_or_video', 'fallback_calls': 0}
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k != 'samples_s'}), flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump({'status': 'pass' if all(r['status'] == 'pass' for r in rows) else 'fail', 'gpu': torch.cuda.get_device_name(),
                   'records': rows, 'warmup': 5, 'repeats': 30, 'video_promotion': False}, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
