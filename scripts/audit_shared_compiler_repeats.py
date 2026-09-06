#!/usr/bin/env python3
"""Audit every repeated video, including preserved before-load failed attempts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_system_video_comparison import compare


def read(path):
    return json.loads(Path(path).read_text())


def audit(original, recoveries):
    manifests = [(original, read(original/'manifest.json'))]
    manifests.extend((p, read(p/'manifest.json')) for p in recoveries)
    original_sha = hashlib.sha256((original/'manifest.json').read_bytes()).hexdigest()
    attempts, final_runs = [], {}
    for root, manifest in manifests:
        if root != original and manifest['original_manifest_sha256'] != original_sha:
            raise ValueError('recovery targets a different frozen batch')
        for row in manifest['cases']:
            run = Path(row['run'])
            execution = read(run/'execution.json')
            if execution['suite_sha256'] != hashlib.sha256(Path(row['suite']).read_bytes()).hexdigest():
                raise ValueError('suite changed')
            if any(execution[k] != row[k] for k in ('lane', 'repeat', 'variant', 'source_commit', 'cpu_ids')):
                raise ValueError('execution differs from frozen manifest')
            identifier = (row['lane'], row['repeat'])
            attempt = {**row, 'status': execution['status'], 'process_wall_s': execution['process_wall_s'],
                       'start_unix': execution['start_unix'], 'execution_sha256': hashlib.sha256((run/'execution.json').read_bytes()).hexdigest()}
            attempts.append(attempt)
            if execution['exit_code'] == 0:
                if identifier in final_runs:
                    raise ValueError('successful case was submitted more than once')
                final_runs[identifier] = attempt
            else:
                log = (run/'runner.log').read_text()
                if 'no idle unlocked local GPU' not in log or 'RUNTIME ' in log or list(run.glob('*/case_state.json')):
                    raise ValueError('non-idle failure requires separate investigation')
    expected = {(r['lane'], r['repeat']) for r in manifests[0][1]['cases']}
    if set(final_runs) != expected:
        raise ValueError(f'GPU executions remain missing: {expected-set(final_runs)}')
    groups = []
    for lane in (0, 1):
        runs = [r for key, r in sorted(final_runs.items()) if key[0] == lane]
        anchor = next(r for r in runs if r['variant'] == 'old')
        records = []
        for row in runs:
            log = (Path(row['run'])/'runner.log').read_text()
            locked = re.search(r'\[gpu\] physical=(\d+)', log)
            if locked is None or int(locked.group(1)) != lane:
                raise ValueError('same physical GPU lock not proven')
            cpus = ','.join(map(str, row['cpu_ids']))
            if f"'taskset', '-c', '{cpus}'" not in log:
                raise ValueError('CPU affinity launch not proven')
            runtime = [json.loads(line[len('RUNTIME '):]) for line in log.splitlines() if line.startswith('RUNTIME ')]
            if len(runtime) != 1:
                raise ValueError('one runtime required per independent process')
            pair = compare(anchor['run'], row['run'])
            if pair['status'] != 'pass' or not pair['bitwise_equal_latents']:
                raise ValueError('route/latent equivalence gate failed')
            if pair['rows'][0]['video_sha256'] != pair['rows'][1]['video_sha256']:
                raise ValueError('video bytes differ')
            if pair['rows'][0]['transferred_bytes'] != pair['rows'][1]['transferred_bytes']:
                raise ValueError('H2D payload differs')
            records.append({**row, **pair['rows'][1], 'gpu': runtime[0],
                            'same_ordered_routes': True, 'same_latent_video_bytes': True})
        chron = sorted(records, key=lambda r: r['start_unix'])
        by_variant = {v: [r for r in records if r['variant'] == v] for v in ('old', 'new')}
        if any(len(v) != 2 for v in by_variant.values()):
            raise ValueError('exactly two repetitions per source required')
        summary = {}
        fields = ('end_to_end_s', 'model_load_s_total', 'materialize_total_s', 'backend_complete_s')
        components = ('routing_s', 'cpu_gather_s', 'h2d_s', 'attention_s', 'q_summary_s', 'rope_s', 'index_s')
        for name in (*fields, *components):
            values = {v: [r[name] if name in fields else r['timing'][name] for r in rows]
                      for v, rows in by_variant.items()}
            med = {v: statistics.median(rows) for v, rows in values.items()}
            summary[name] = {'old_samples': values['old'], 'new_samples': values['new'],
                'old_median': med['old'], 'new_median': med['new'],
                'reduction': 1-med['new']/med['old'] if med['old'] else None}
        order = ''.join('A' if r['variant'] == 'old' else 'B' for r in chron)
        groups.append({'prompt': records[0]['prompt'], 'lane': lane, 'records': records,
            'actual_successful_order': order, 'original_counterbalanced_order_preserved': order in ('ABBA', 'BAAB'),
            'samples_per_source': 2, 'summary': summary,
            'all_raw_components_nonregressing': all(summary[k]['reduction'] >= 0 for k in components),
            'component_observations_not_additive_critical_path': True,
            'inferential_CI_not_claimed_with_n2': True})
    return {'status': 'pass', 'technical_successes': len(final_runs),
        'preserved_preload_failures': sum(r['status'] != 'pass' for r in attempts), 'missing': 0,
        'groups': groups, 'attempts': attempts, 'original_manifest_sha256': original_sha,
        'scope': 'same_route_same_hardware_complete_video_latency_repeats',
        'not_independent_quality_samples': True, 'cross_prompt_absolute_times_not_pooled': True,
        'startup_and_model_load_separate': True, 'formal_quality_promotion': False,
        'absolute_quality_inherited_via_byte_equivalence_not_new_semantic_review': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', required=True)
    parser.add_argument('--recovery', action='append', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = audit(Path(args.original).resolve(), [Path(p).resolve() for p in args.recovery])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': result['status'], 'successes': result['technical_successes'],
        'preserved_failures': result['preserved_preload_failures'], 'missing': result['missing'],
        'groups': [{'prompt': g['prompt'], 'order': g['actual_successful_order'], 'summary': g['summary']}
                   for g in result['groups']]}, indent=2))


if __name__ == '__main__':
    main()
