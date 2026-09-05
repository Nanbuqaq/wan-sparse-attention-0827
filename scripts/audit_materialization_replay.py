#!/usr/bin/env python3
"""Bounded real-capture CPU experiment; no model generation or GPU reservation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.selectors import SparseSelection
from adapters.longlive_sparse.stats import TimingBreakdown
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer
from adapters.longlive_sparse.transfer_plan import build_transfer_plan


def process_snapshot(pid: int) -> dict:
    path = Path(f'/proc/{pid}/status')
    if not path.exists():
        return {'pid': pid, 'present': False}
    fields = {}
    for line in path.read_text().splitlines():
        name, _, value = line.partition(':')
        if name in {'Threads', 'VmRSS', 'State', 'Cpus_allowed_list'}:
            fields[name] = value.strip()
    return {'pid': pid, **fields}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, nargs='+', default=[1, 4, 16, 128])
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--observe-pids', type=int, nargs='*', default=[])
    parser.add_argument('--plan-replay', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    capture = torch.load(args.capture, map_location='cpu', weights_only=True)
    key, value = capture['key'], capture['value']
    frames, tokens = capture['frame_ids'].long(), capture['token_ids'].long()
    ids = list(dict.fromkeys(frames[0, 0].tolist()))
    archive = HistoryArchive(SparseHistoryConfig(method='transfer_vaware_hybrid_history' if args.plan_replay else 'rag_dense'),
                             spatial_height=30, spatial_width=52)
    for frame in ids:
        mask = frames[0, 0] == frame
        archive.index_frame(0, frame, key[:, mask], value[:, mask])
    selection = SparseSelection(frames, tokens, torch.zeros_like(frames),
        key.shape[1], key.shape[1], key.shape[1], key.shape[1], None, None, 0,
        TimingBreakdown())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = {
        'status': 'running', 'scope': 'CPU-only original real-capture Dense materialization',
        'capture': args.capture, 'shape': list(key.shape), 'dtype': str(key.dtype),
        'source_stride': list(key.stride()),
        'fixed_head_stride': list(key[0, :, 0].stride()),
        'fixed_head_contiguous': key[0, :, 0].is_contiguous(),
        'cpu_quota_us': Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text().strip(),
        'cpu_period_us': Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text().strip(),
        'background': [process_snapshot(pid) for pid in args.observe_pids], 'rows': [],
    }
    plans = {}
    if args.plan_replay:
        for method in ('rag_dense', 'transfer_vaware_hybrid_history'):
            params = dict(base_fraction=.7, local_fraction=.15, v_weight=1., transfer_multiplier=1.) if method != 'rag_dense' else {}
            archive.config = SparseHistoryConfig(method=method, history_density=1. if method == 'rag_dense' else .25,
                                                 method_params=params)
            if method == 'rag_dense':
                route = archive.full_history_route(0, ids, query_shape=capture['query'].shape, exact_k_tokens=0)
            else:
                route = archive.route_indexed(0, summarize_query_for_pretransfer(capture['query'], 64), ids, exact_k_tokens=0)
            plans[method] = (route, build_transfer_plan(route, ids, frame_tokens=1560,
                layout='exact_compact', bytes_per_token=2*key.shape[-1]*key.element_size()))
    for threads in args.threads:
        torch.set_num_threads(threads)
        if args.plan_replay:
            for method, (route, transfer) in plans.items():
                expected = None
                for policy in ['candidate_gather', 'archive_runs']:
                    samples = []
                    for _ in range(args.iterations):
                        start = time.perf_counter()
                        mat = archive.materialize_transfer_plan(0, transfer, route, device='cpu',
                            current_frame_id=120, freqs=None, cpu_pack_policy=policy)
                        samples.append({'complete_s': time.perf_counter()-start,
                            **{name: getattr(mat, name) for name in ('cpu_prepare_s', 'cpu_pack_s',
                                'cpu_allocate_pin_s', 'gpu_restore_s', 'materialize_total_s')}})
                    if expected is None:
                        expected = mat
                    assert torch.equal(expected.key, mat.key) and torch.equal(expected.value, mat.value)
                    row = {'method': method, 'policy': policy, 'threads': threads, 'kv_exact': True,
                        'route_sha': route.digest(), 'samples': samples,
                        'median_complete_s': statistics.median(item['complete_s'] for item in samples)}
                    snapshot['rows'].append(row)
                    output.write_text(json.dumps(snapshot, indent=2) + '\n')
                    print(json.dumps({name: row[name] for name in ('method', 'policy', 'threads', 'median_complete_s')}), flush=True)
            continue
        samples = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            candidate = archive.dense_history_tensors(0, ids)
            concat_s = time.perf_counter() - start
            start = time.perf_counter()
            materialized = archive.materialize(0, selection, device='cpu',
                current_frame_id=120, freqs=None, candidate_frame_ids=ids,
                dense_key=candidate[0], dense_value=candidate[1],
                dense_frame_ids=candidate[2], dense_token_ids=candidate[3])
            materialize_s = time.perf_counter() - start
            samples.append({'concat_s': concat_s, 'materialize_s': materialize_s,
                            'complete_s': concat_s + materialize_s,
                            'reported_cpu_gather_s': materialized.cpu_gather_s})
        # Validation outside measured intervals; original captured order matches archive IDs.
        assert torch.equal(materialized.key, key)
        assert torch.equal(materialized.value, value)
        row = {'threads': threads, 'process': process_snapshot(os.getpid()),
               'samples': samples, 'median_complete_s': statistics.median(
                   sample['complete_s'] for sample in samples), 'kv_exact': True}
        snapshot['rows'].append(row)
        output.write_text(json.dumps(snapshot, indent=2) + '\n')
        print(json.dumps({'threads': threads, 'median_complete_s': row['median_complete_s']}), flush=True)
    snapshot['status'] = 'pass'
    output.write_text(json.dumps(snapshot, indent=2) + '\n')


if __name__ == '__main__':
    main()
