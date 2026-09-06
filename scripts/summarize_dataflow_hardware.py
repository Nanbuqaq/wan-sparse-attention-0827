#!/usr/bin/env python3
"""Normalize within each physical GPU; never pool absolute hardware times."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def summarize(root, expected_indices):
    cases = [json.loads(p.read_text()) for p in sorted(root.glob('lane*/case[0-9][0-9].json'))]
    if {c['case']['index'] for c in cases} != set(expected_indices) or len(cases) != len(expected_indices):
        raise ValueError(f'incomplete/duplicate matrix: {root}')
    names = {c['gpu'] for c in cases}
    if len(names) != 1:
        raise ValueError('mixed GPU labels inside one hardware summary')
    winners, records, locks = defaultdict(Counter), [], []
    for c in cases:
        if c['status'] != 'pass' or c['warmup'] != 5 or c['repeats'] != 30 or len(c['records']) != 10:
            raise ValueError('full frozen per-case measurement protocol required')
        scopes = defaultdict(list)
        for r in c['records']:
            if len(r['samples']) != 30:
                raise ValueError('thirty measured repeats required')
            scopes[r['scope']].append(r)
        for scope, rows in scopes.items():
            winner = min(rows, key=lambda r: r['wall_ms_median'])
            winners[scope][winner['backend']] += 1
            baseline_name = 'grouped_fa2' if scope in ('resident', 'onload_inclusive') else 'qout_gpu_page_reference'
            base = next(r for r in rows if r['backend'] == baseline_name)
            for r in rows:
                records.append({'gpu': c['gpu'], 'case_index': c['case']['index'], **c['case'],
                    'scope': scope, 'backend': r['backend'], 'wall_ms_median': r['wall_ms_median'],
                    'wall_ms_p95': r['wall_ms_p95'], 'normalized_speedup': base['wall_ms_median']/r['wall_ms_median'],
                    'normalizer_backend': baseline_name, 'route_sha256': c['route_sha256'],
                    'kv_partial_workspace_bytes': c['kv_partial_workspace_bytes']})
    for p in sorted(root.glob('lane*/case[0-9][0-9].json')):
        locks.append({'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    return {'status': 'pass', 'gpu': next(iter(names)), 'cases': len(cases), 'missing': 0,
        'winner_counts': {s: dict(v) for s, v in winners.items()}, 'records': records, 'sources': locks,
        'prepared_FA2_not_production_grouped_wrapper': True,
        'reference_KVOut_not_tuned_state_of_the_art': True,
        'actual_HBM_transactions_not_inferred_from_workspace_or_runtime': True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--boundary', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    points = json.loads((Path(__file__).resolve().parents[1]/'configs/system/dataflow_boundary_points.json').read_text())['case_indices']
    full = summarize(args.matrix, range(72))
    hardware = [full] + [summarize(root, points) for root in args.boundary]
    for item in hardware[1:]:
        routes = {r['case_index']: r['route_sha256'] for r in item['records']}
        reference = {r['case_index']: r['route_sha256'] for r in full['records']}
        if any(reference[i] != digest for i, digest in routes.items()):
            raise ValueError('cross-hardware geometry/route mismatch')
    result = {'status': 'pass', 'hardware': hardware, 'common_boundary_indices': points,
        'absolute_times_never_pooled': True, 'no_video_or_optimized_adaptive_promotion': True}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    records = [r for h in hardware for r in h['records']]
    with (args.output/'points.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, scope in zip(axes, ('resident', 'onload_inclusive')):
        for item in hardware:
            values = [next(r['normalized_speedup'] for r in item['records'] if r['case_index'] == index
                          and r['scope'] == scope and r['backend'] == 'kv_stationary_split_reference') for index in points]
            ax.plot(range(len(points)), values, marker='o', label=item['gpu'])
        ax.axhline(1., color='grey', linestyle='--')
        ax.set_xticks(range(len(points)), points)
        ax.set(xlabel='Frozen boundary case index', ylabel='KV reference speedup vs prepared FA2', title=scope)
        ax.legend(fontsize=8)
    fig.savefig(args.output/'normalized_boundary.png', dpi=150)
    plt.close(fig)
    print(json.dumps({'status': 'pass', 'hardware': [{k:v for k,v in h.items() if k not in ('records', 'sources')} for h in hardware]}))


if __name__ == '__main__':
    main()
