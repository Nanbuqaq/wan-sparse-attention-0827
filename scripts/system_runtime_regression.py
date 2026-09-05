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
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ['LONGLIVE_CAPTURE_COMPLETE_ATTENTION'] = '1'
    os.environ['LONGLIVE_COMPLETE_CAPTURE_LAYERS'] = '0'
    os.environ['LONGLIVE_COMPLETE_CAPTURE_STARTS'] = '640'
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
    query = torch.randn(1, 128, 128, device=device, dtype=torch.bfloat16)
    history = [(torch.randn(1, 128, 2, 64, dtype=torch.bfloat16),
                torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)) for _ in range(2)]
    kv = {'k': torch.randn(1, 768, 2, 64, device=device, dtype=torch.bfloat16),
          'v': torch.randn(1, 768, 2, 64, device=device, dtype=torch.bfloat16),
          'global_end_index': torch.tensor([640], device=device),
          'local_end_index': torch.tensor([384], device=device)}
    freqs = rope_params(1024, 64).to(device)
    grid = torch.tensor([[1, 8, 16]], device=device)
    final_params = dict(base_fraction=.7, local_fraction=.15, v_weight=1., transfer_multiplier=1.)
    records, reference_outputs, reference_shas = [], {}, {}
    shared_weights = None
    # Each system benefit is measured against the same method and weights.
    for method in ['rag_dense', 'block64_history', 'transfer_vaware_hybrid_history', 'system_utility_history']:
        params = final_params if method == 'transfer_vaware_hybrid_history' else (
            {'value_candidate': 'peak_value', 'cost_strategy': 'static_block'}
            if method == 'system_utility_history' else {})
        for policy, cache_enabled in [('legacy', False), ('candidate_gather', False),
                                      ('archive_runs', False), ('archive_runs', True)]:
            config = SparseHistoryConfig(method=method, history_density=1. if method == 'rag_dense' else .25,
                refresh_policy='per_chunk', rope_policy='upstream_zero', method_params=params)
            archive = HistoryArchive(config, spatial_height=8, spatial_width=16)
            for frame, (key, value) in enumerate(history, 1):
                archive.index_frame(0, frame, key, value)
            system = LongLiveSystemConfig(transfer_layout='legacy' if policy == 'legacy' else 'exact_compact',
                cpu_pack_policy='candidate_gather' if policy == 'legacy' else policy,
                staging_mode='persistent_fused' if policy != 'legacy' else 'per_call_separate',
                gpu_union_cache='per_chunk' if cache_enabled else 'off',
                gpu_union_cache_budget_mib=16 if cache_enabled else 0)
            cache = HistoryUnionCache(16 * 1024**2) if cache_enabled else None
            pool = PinnedStagingPool(slots=2, budget_bytes=16*1024**2, pin_memory=True)
            module = SparseHistorySelfAttention(dim=128, num_heads=2, local_attn_size=6,
                sink_size=1, memory_size=2, layer_id=0, history_archive=archive,
                sparse_config=config, system_config=system,
                history_union_cache=cache, history_staging_pool=pool).to(device, dtype=torch.bfloat16)
            module.max_attention_size = 768
            if shared_weights is None:
                shared_weights = {key: value.clone() for key, value in module.state_dict().items()}
            module.load_state_dict(shared_weights)
            def forward():
                return module(query, torch.tensor([128], device=device), grid, freqs, None,
                              kv_cache=kv, current_start=640, memory_indices=torch.tensor([[0, 1]], device=device))[0]
            os.environ['LONGLIVE_CAPTURE_CASE_TAG'] = f'{method}_{policy}_{cache_enabled}'
            forward()  # warmup/capture outside measurement
            capture_path = module._capture_root('complete_attention_captures') / 'layer00_start00000640_pass00.pt'
            captured = torch.load(capture_path, map_location='cpu', weights_only=True)
            assert captured['contains_sink_current_recent'] and captured['scope'].endswith('post_rope')
            if method == 'rag_dense':
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
                start = time.perf_counter()
                outputs = [forward() for _ in range(5)]
                torch.cuda.synchronize(device)
                times.append(time.perf_counter() - start)
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
                      'five_call_wall_samples_s': times, 'five_call_wall_median_s': statistics.median(times),
                      'route_sha': route_shas, 'max_abs_vs_same_method_legacy': delta,
                      'cache': cache.as_dict() if cache else None, 'status': 'pass'}
            records.append(record)
            print(json.dumps(record), flush=True)
    output_path.write_text(json.dumps({'status': 'pass', 'scope': 'synthetic real CUDA runtime self-attention forward',
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'rows': records,
        'video_quality_claim': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
