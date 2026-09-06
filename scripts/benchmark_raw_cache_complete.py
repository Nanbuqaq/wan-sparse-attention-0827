#!/usr/bin/env python3
"""Fixed-route cold/warm raw-cache cost gate, including CPU preparation/restore.

Compared with uncached contiguous-run materialization, NOT with the much cheaper
per-chunk union hit. No RoPE/Attention/VAE or video speed claim is made here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.history_cache import RawHistoryBlockCache
from adapters.longlive_sparse.route_plan import HistoryRoutePlan, map_union_coordinates
from adapters.longlive_sparse.staging import PinnedStagingPool
from adapters.longlive_sparse.transfer_plan import build_transfer_plan


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--kind', choices=('motion', 'state'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('real GPU required')
    workspace = Path(args.workspace).resolve()
    root = workspace/'results/metrics/matched_trajectory_capture_be00491'
    audit = json.loads((root/'trajectory_audit.json').read_text())
    actor = next(r for r in audit['cases'] if r['method'] == 'transfer_vaware_hybrid_history'
                 and r['prompt'] == f'calibration_{args.kind}')
    path = Path(actor['capture_dir'])/'layer09_start00046800_pass00.pt'
    capture = torch.load(path, map_location='cpu', weights_only=True)
    route = HistoryRoutePlan.from_state_dict(capture['route_plan'])
    if capture['key'].shape[1] != 9360 or route.digest() != capture['route_sha']:
        raise ValueError('audited full six-frame history required')
    start = time.perf_counter()
    archive = HistoryArchive(SparseHistoryConfig(method='block64_history', history_density=.25),
                             spatial_height=30, spatial_width=52)
    candidates = list(dict.fromkeys(capture['frame_ids'][0, 0].tolist()))
    for frame in candidates:
        selected = torch.nonzero(capture['frame_ids'][0, 0] == frame).flatten()
        selected = selected[capture['token_ids'][0, 0, selected].argsort()]
        k = capture['key_unrotated'].index_select(1, selected).clone()
        v = capture['value'].index_select(1, selected).clone()
        archive.index_frame(9, frame, k, v)
    archive_prepare_s = time.perf_counter()-start
    indices = map_union_coordinates(route, capture['frame_ids'], capture['token_ids'])
    expected = []
    for name in ('key_unrotated', 'value'):
        source = capture[name].permute(0, 2, 1, 3)
        expected.append(source.gather(2, indices[..., None].expand(-1, -1, -1, source.shape[-1]))
                        .permute(0, 2, 1, 3).cuda())
    transfer = build_transfer_plan(route, candidates, frame_tokens=1560, layout='exact_compact', bytes_per_token=512)
    pool = PinnedStagingPool(slots=2, budget_bytes=128*1024**2, pin_memory=True)
    records = []
    for repeat in range(4):  # First round warmup; three paired reported rounds.
        cache = RawHistoryBlockCache(64*1024**2)
        order = ('archive_runs', 'raw_cold', 'raw_warm') if repeat % 2 == 0 else ('raw_cold', 'raw_warm', 'archive_runs')
        for mode in order:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            if mode == 'archive_runs':
                result = archive.materialize_transfer_plan(9, transfer, route, device='cuda', current_frame_id=30,
                    freqs=None, staging_pool=pool, staging_mode='persistent_separate', cpu_pack_policy='archive_runs')
            else:
                result = archive.materialize_raw_block_cached(9, route, cache, device='cuda', current_frame_id=30,
                    freqs=None, candidate_frame_ids=candidates)
            torch.cuda.synchronize()
            wall_s = time.perf_counter()-start
            if not torch.equal(result.key_unrotated, expected[0]) or not torch.equal(result.value, expected[1]):
                raise ValueError('raw immutable KV or logical union order changed')
            if mode == 'raw_warm' and result.transferred_bytes != 0:
                raise ValueError('fixed-route warm cache missed')
            records.append({'repeat': repeat, 'warmup': repeat == 0, 'mode': mode, 'complete_s': wall_s,
                'materialize_total_s': result.materialize_total_s, 'cpu_prepare_s': result.cpu_prepare_s,
                'cpu_gather_s': result.cpu_gather_s, 'h2d_s': result.h2d_s,
                'gpu_restore_s': result.gpu_restore_s, 'transferred_bytes': result.transferred_bytes,
                'payload_bytes': result.payload_bytes, 'padding_bytes': result.padding_bytes,
                'h2d_copy_count': result.h2d_copy_count, 'cache_hit_bytes': result.cache_hit_bytes,
                'peak_allocated_gpu_bytes': torch.cuda.max_memory_allocated(), 'bitwise_raw_kv': True})
            print(json.dumps(records[-1]), flush=True)
            del result
    summary = {mode: {field: statistics.median(r[field] for r in records if r['mode'] == mode and not r['warmup'])
        for field in ('complete_s', 'cpu_prepare_s', 'cpu_gather_s', 'h2d_s', 'gpu_restore_s', 'transferred_bytes')}
        for mode in ('archive_runs', 'raw_cold', 'raw_warm')}
    result = {'status': 'pass', 'outcome': 'negative_for_unmodified_raw_cache_runtime'
        if summary['raw_warm']['complete_s'] > summary['archive_runs']['complete_s'] else 'component_gate_only',
        'gpu': torch.cuda.get_device_name(), 'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'capture_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'route_sha256': route.digest(),
        'archive_prepare_s_separate': archive_prepare_s, 'records': records, 'summary': summary,
        'scope': 'fixed_route_materialization_complete_wall_no_RoPE_Attention_VAE',
        'real_gpu_restore_and_copies': True, 'formal_promotion': False,
        'not_a_comparison_against_per_chunk_union_hit': True, 'no_cross_chunk_hit_rate_claim': True}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
