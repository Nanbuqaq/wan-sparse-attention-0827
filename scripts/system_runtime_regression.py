#!/usr/bin/env python3
"""Real CUDA self-attention forward regression before any video batch.

Synthetic inputs test branch correctness, not video quality. Times include the
entire self-attention forward, route/plan/cache, copies, RoPE and output projection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import os
import subprocess
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from adapters.longlive_sparse.history_cache import HistoryUnionCache
from adapters.longlive_sparse.staging import PinnedStagingPool


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--large', action='store_true', help='LongLive Q=4680, history=9360, H=12, D=128')
    parser.add_argument('--method-filter')
    parser.add_argument('--value-candidate', default='peak_value')
    parser.add_argument('--group-top-p', type=float, default=0.)
    parser.add_argument('--profile-policy', choices=('legacy', 'candidate_gather', 'archive_runs', 'cache'),
                        help='Nsight cudaProfilerApi: only the first measured five-call window of this policy')
    args = parser.parse_args()
    if args.profile_policy:
        os.environ['LONGLIVE_NVTX'] = '1'
        if not args.method_filter:
            parser.error('--profile-policy requires --method-filter to bound the trace')
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ['LONGLIVE_CAPTURE_COMPLETE_ATTENTION'] = '1'
    os.environ['LONGLIVE_COMPLETE_CAPTURE_LAYERS'] = '0'
    frame_tokens, heads, head_dim = (1560, 12, 128) if args.large else (128, 2, 64)
    new_frames, history_frames, local_frames, previous_local = (3, 6, 12, 9) if args.large else (1, 2, 6, 3)
    current_frame = 20 if args.large else 5
    current_start = current_frame * frame_tokens
    model_dim = heads * head_dim
    os.environ['LONGLIVE_COMPLETE_CAPTURE_STARTS'] = str(current_start)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('real CUDA required')
    from adapters.longlive_sparse.runtime_attention import SparseHistorySelfAttention
    from wan.modules.model import rope_params
    from adapters.longlive_sparse.offline_eval import dense_history_attention, routed_history_attention
    from adapters.longlive_sparse.route_plan import HistoryRoutePlan

    device = torch.device('cuda:0')
    torch.manual_seed(20260911)
    query = torch.randn(1, new_frames * frame_tokens, model_dim, device=device, dtype=torch.bfloat16)
    history = [(torch.randn(1, frame_tokens, heads, head_dim, dtype=torch.bfloat16),
                torch.randn(1, frame_tokens, heads, head_dim, dtype=torch.bfloat16)) for _ in range(history_frames)]
    kv = {'k': torch.randn(1, local_frames * frame_tokens, heads, head_dim, device=device, dtype=torch.bfloat16),
          'v': torch.randn(1, local_frames * frame_tokens, heads, head_dim, device=device, dtype=torch.bfloat16),
          'global_end_index': torch.tensor([current_start], device=device),
          'local_end_index': torch.tensor([previous_local * frame_tokens], device=device)}
    freqs = rope_params(1024, head_dim).to(device)
    height, width = (30, 52) if args.large else (8, 16)
    grid = torch.tensor([[new_frames, height, width]], device=device)
    final_params = dict(base_fraction=.7, local_fraction=.15, v_weight=1., transfer_multiplier=1.)
    records, reference_outputs, reference_shas = [], {}, {}
    shared_weights = None
    # Each system benefit is measured against the same method and weights.
    for method in ['rag_dense', 'block64_history', 'transfer_vaware_hybrid_history', 'system_utility_history']:
        if args.method_filter and method != args.method_filter:
            continue
        params = final_params if method == 'transfer_vaware_hybrid_history' else (
            {'value_candidate': args.value_candidate, 'cost_strategy': 'static_block'}
            if method == 'system_utility_history' else {})
        for policy, cache_enabled in [('legacy', False), ('candidate_gather', False),
                                      ('archive_runs', False), ('archive_runs', True)]:
            config = SparseHistoryConfig(method=method, history_density=1. if method == 'rag_dense' else .25,
                refresh_policy='per_chunk', rope_policy='upstream_zero', method_params=params)
            archive = HistoryArchive(config, spatial_height=height, spatial_width=width)
            for frame, (key, value) in enumerate(history, 1):
                archive.index_frame(0, frame, key, value)
            system = LongLiveSystemConfig(transfer_layout='legacy' if policy == 'legacy' else 'exact_compact',
                cpu_pack_policy='candidate_gather' if policy == 'legacy' else policy,
                staging_mode='persistent_fused' if policy != 'legacy' else 'per_call_separate',
                gpu_union_cache='per_chunk' if cache_enabled else 'off',
                group_selection_policy='mass_preserving_top_p' if args.group_top_p else 'legacy_exact_union',
                group_top_p=args.group_top_p or .90,
                gpu_union_cache_budget_mib=256 if cache_enabled else 0)
            cache = HistoryUnionCache(256 * 1024**2) if cache_enabled else None
            pool = PinnedStagingPool(slots=2, budget_bytes=256*1024**2, pin_memory=True)
            module = SparseHistorySelfAttention(dim=model_dim, num_heads=heads, local_attn_size=local_frames,
                sink_size=1, memory_size=history_frames, layer_id=0, history_archive=archive,
                sparse_config=config, system_config=system,
                history_union_cache=cache, history_staging_pool=pool).to(device, dtype=torch.bfloat16)
            module.max_attention_size = local_frames * frame_tokens
            if shared_weights is None:
                shared_weights = {key: value.clone() for key, value in module.state_dict().items()}
            module.load_state_dict(shared_weights)
            torch.cuda.reset_peak_memory_stats()
            def forward():
                return module(query, torch.tensor([query.shape[1]], device=device), grid, freqs, None,
                              kv_cache=kv, current_start=current_start,
                              memory_indices=torch.arange(history_frames, device=device).view(1,-1))[0]
            os.environ['LONGLIVE_CAPTURE_CASE_TAG'] = f'{method}_{policy}_{cache_enabled}'
            capture_enabled = not args.large or (method == 'rag_dense' and policy == 'legacy')
            os.environ['LONGLIVE_CAPTURE_COMPLETE_ATTENTION'] = '1' if capture_enabled else '0'
            forward()  # warmup/capture outside measurement
            if capture_enabled:
                capture_path = module._capture_root('complete_attention_captures') / f'layer00_start{current_start:08d}_pass00.pt'
                captured = torch.load(capture_path, map_location='cpu', weights_only=True)
                assert captured['contains_sink_current_recent'] and captured['scope'].endswith('post_rope')
            if method == 'rag_dense' and not args.large:
                captured_plan = HistoryRoutePlan.from_state_dict(captured['route_plan'])
                full_teacher = dense_history_attention(captured['query'],
                    torch.cat((captured['exact_key'], captured['key']), dim=1),
                    torch.cat((captured['exact_value'], captured['value']), dim=1))
                replay = routed_history_attention(captured['query'], captured['key'], captured['value'],
                    captured['frame_ids'], captured['token_ids'], captured_plan,
                    exact_key=captured['exact_key'], exact_value=captured['exact_value'])
                torch.testing.assert_close(replay, full_teacher, atol=1e-6, rtol=1e-6)
            times = []
            for repeat in range(args.repeats):
                module.clear_selection_cache()
                if cache is not None:
                    cache.reset()
                torch.cuda.synchronize(device)
                profiling_window = (args.profile_policy == ('cache' if cache_enabled else policy) and repeat == 0)
                if profiling_window:
                    torch.cuda.cudart().cudaProfilerStart()
                start = time.perf_counter()
                outputs = [forward() for _ in range(5)]
                torch.cuda.synchronize(device)
                times.append(time.perf_counter() - start)
                if profiling_window:
                    torch.cuda.cudart().cudaProfilerStop()
            output = outputs[-1]
            route_shas = sorted(archive.stats.route_plan_sha256_counts)
            if policy == 'legacy':
                reference_outputs[method] = output
                reference_shas[method] = route_shas
            delta = float((reference_outputs[method].float() - output.float()).abs().max())
            assert delta == 0, (method, policy, delta)
            assert route_shas == reference_shas[method], (method, policy, 'route changed')
            if cache is not None:
                assert (cache.hits, cache.misses) == (4, 1), cache.as_dict()
            record = {'method': method, 'cpu_pack_policy': policy, 'cache_enabled': cache_enabled,
                      'value_candidate': args.value_candidate if method == 'system_utility_history' else None,
                      'group_top_p': args.group_top_p,
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'executor_storage_estimate': archive.stats.call_records[-1].get('grouped_executor_storage'),
                      'five_call_wall_samples_s': times, 'five_call_wall_median_s': statistics.median(times),
                      'route_sha': route_shas, 'max_abs_vs_same_method_legacy': delta,
                      'cache': cache.as_dict() if cache else None, 'status': 'pass'}
            records.append(record)
            (output_path.parent/f'{method}_{policy}_{cache_enabled}_stats.json').write_text(
                json.dumps(archive.stats.as_dict(), indent=2) + '\n')
            print(json.dumps(record), flush=True)
    output_path.write_text(json.dumps({'status': 'pass', 'scope': 'synthetic real CUDA runtime self-attention forward',
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'rows': records,
        'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'qkv_shape': [1, query.shape[1], history_frames * frame_tokens, heads, head_dim],
        'timing_scope': 'profiled_diagnostic' if args.profile_policy else 'unprofiled_wall',
        'video_quality_claim': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
