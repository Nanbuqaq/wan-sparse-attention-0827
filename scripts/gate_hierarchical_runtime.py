#!/usr/bin/env python3
"""Two real chunks, five calls each: union-only vs hierarchical raw residency."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.history_cache import HistoryUnionCache, HierarchicalHistoryCache
from adapters.longlive_sparse.offload import ArchiveOffloadStager
from adapters.longlive_sparse.offline_eval import dense_history_attention, output_error_metrics
from adapters.longlive_sparse.phase_prototypes import canonical_wan_frequency_table
from adapters.longlive_sparse.staging import PinnedStagingPool
from adapters.longlive_sparse.system_config import LongLiveSystemConfig


@torch.inference_mode()
def run(large, method='transfer_vaware_hybrid_history'):
    import adapters.longlive_sparse.runtime_attention as runtime
    torch.manual_seed(20260907)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    height, width, heads, dim, local, history, new = (30, 52, 12, 128, 12, 6, 3) if large else (8, 16, 2, 64, 6, 2, 1)
    tokens = height*width
    params = {'base_fraction': .7, 'local_fraction': .15, 'v_weight': 1., 'transfer_multiplier': 1., 'query_block_size': 64}
    cfg = SparseHistoryConfig(method=method, history_density=1. if method == 'rag_dense' else .25,
                               refresh_policy='per_chunk', method_params={} if method == 'rag_dense' else params)
    original = [(torch.randn(1, tokens, heads, dim, dtype=torch.bfloat16),
                 torch.randn(1, tokens, heads, dim, dtype=torch.bfloat16)) for _ in range(history)]
    initial_k = torch.randn(1, local*tokens, heads, dim, device='cuda', dtype=torch.bfloat16)
    initial_v = torch.randn_like(initial_k)
    inputs = [torch.randn(1, new*tokens, heads*dim, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    freqs = canonical_wan_frequency_table(dim).cuda()
    outputs, snapshots, stats, cache_stats = {}, {}, {}, {}
    weights = None
    reference_execute = runtime.execute_plan
    for mode in ('per_chunk', 'hierarchical'):
        archive = HistoryArchive(cfg, spatial_height=height, spatial_width=width)
        for frame, (k, v) in enumerate(original, 1):
            archive.index_frame(0, frame, k.cuda(), v, storage_k=k, storage_v=v)
        system = LongLiveSystemConfig(transfer_layout='exact_compact', cpu_pack_policy='archive_runs',
            staging_mode='persistent_separate', archive_offload='pooled_pageable', host_pinned_budget_mib=128,
            gpu_union_cache=mode, gpu_union_cache_budget_mib=256,
            raw_cache_budget_mib=128 if mode == 'hierarchical' else 0)
        cache = HierarchicalHistoryCache(256*1024**2, 128*1024**2) if mode == 'hierarchical' else HistoryUnionCache(256*1024**2)
        pool = PinnedStagingPool(slots=2, budget_bytes=128*1024**2, pin_memory=True)
        module = runtime.SparseHistorySelfAttention(dim=heads*dim, num_heads=heads, local_attn_size=local,
            sink_size=1, memory_size=history, layer_id=0, history_archive=archive, sparse_config=cfg,
            system_config=system, history_union_cache=cache, history_staging_pool=pool).cuda().bfloat16()
        module.archive_offload_stager = ArchiveOffloadStager(pool)
        module.max_attention_size = local*tokens
        if weights is None:
            weights = {k: v.clone() for k, v in module.state_dict().items()}
        module.load_state_dict(weights)
        start = (history+local)*tokens
        kv = {'k': initial_k.clone(), 'v': initial_v.clone(), 'global_end_index': torch.tensor([start], device='cuda'),
            'local_end_index': torch.tensor([local*tokens], device='cuda'),
            'cpu_k_frames': [k.unsqueeze(1) for k, v in original], 'cpu_v_frames': [v.unsqueeze(1) for k, v in original]}
        calls = []
        def audited_execute(backend, q, ek, ev, hk, hv, plan):
            result = reference_execute(backend, q, ek, ev, hk, hv, plan)
            teacher = dense_history_attention(q, torch.cat((ek, hk), 1), torch.cat((ev, hv), 1))
            error = output_error_metrics(teacher, result.output)
            if error['max_abs'] > .02 or error['relative_l2'] > .01 or error['one_minus_cosine'] > .001:
                raise RuntimeError(f'BF16 vs full selected-context FP32 gate failed: {error}')
            calls.append({'route_sha': plan.digest(), 'hk': hk.cpu(), 'hv': hv.cpu(), 'error': error})
            return result
        runtime.execute_plan = audited_execute
        rendered = []
        try:
            for chunk in range(2):
                for call in range(5):
                    value, update = module(inputs[chunk]+call*.1, torch.tensor([new*tokens], device='cuda'),
                        torch.tensor([[new, height, width]], device='cuda'), freqs, None, kv_cache=kv,
                        current_start=start+chunk*new*tokens,
                        memory_indices=torch.arange(chunk, chunk+history, device='cuda').view(1, -1))
                    runtime._UPSTREAM.CausalWanModel._apply_cache_updates(None, [kv], [(0, update)])
                    rendered.append(value.cpu())
        finally:
            runtime.execute_plan = reference_execute
        if (cache.hits, cache.misses) != (8, 2):
            raise RuntimeError('five-call union reuse was not preserved across both chunks')
        outputs[mode] = (rendered, calls)
        snapshots[mode] = {frame: (entry.key.clone(), entry.value.clone(), entry.block_centroids.clone(),
                                  entry.block_value_centroids.clone()) for frame, entry in archive._layers[0].items()}
        stats[mode] = archive.stats.as_dict()
        cache_stats[mode] = cache.as_dict()
    for left, right in zip(outputs['per_chunk'][0], outputs['hierarchical'][0]):
        if not torch.equal(left, right):
            raise RuntimeError('complete forward output changed')
    records = []
    for i, (left, right) in enumerate(zip(outputs['per_chunk'][1], outputs['hierarchical'][1])):
        if left['route_sha'] != right['route_sha'] or not torch.equal(left['hk'], right['hk']) or not torch.equal(left['hv'], right['hv']):
            raise RuntimeError('route or original selected RoPE-K/V changed')
        records.append({'call': i, 'route_sha': left['route_sha'], 'bf16_vs_fp32': right['error'], 'same_KV_and_output': True})
    if snapshots['per_chunk'].keys() != snapshots['hierarchical'].keys():
        raise RuntimeError('archive frames diverged')
    for frame, tensors in snapshots['per_chunk'].items():
        if not all(torch.equal(a, b) for a, b in zip(tensors, snapshots['hierarchical'][frame])):
            raise RuntimeError('future archive KV/prototypes changed')
    raw = cache_stats['hierarchical']['raw_slab']
    if large and raw['hit_bytes'] <= 0:
        raise RuntimeError('cross-chunk raw hit branch was not reached')
    if stats['hierarchical']['transferred_bytes'] > stats['per_chunk']['transferred_bytes']:
        raise RuntimeError('raw residency increased KV payload')
    return {'status': 'pass', 'scope': 'two_real_chunks_with_eviction_and_five_call_union_reuse',
        'large': large, 'method': method, 'gpu': torch.cuda.get_device_name(), 'records': records, 'cache': cache_stats,
        'original_and_newly_evicted_archive_KV_prototypes_equal': True,
        'kv_bytes': {k: s['transferred_bytes'] for k, s in stats.items()},
        'index_h2d_bytes': {k: s['restore_index_h2d_bytes'] for k, s in stats.items()},
        'no_video_speed_claim': True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--large', action='store_true')
    p.add_argument('--method', choices=('rag_dense', 'transfer_vaware_hybrid_history'), default='transfer_vaware_hybrid_history')
    args = p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.large, args.method)
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
