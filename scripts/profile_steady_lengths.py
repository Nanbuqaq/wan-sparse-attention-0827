#!/usr/bin/env python3
"""Optimized real generator/VAE at39/120/240, three short Nsight windows.

Load once; independent seeded video trajectories. No captures inside profiled
calls and no artifact encoding. Recorded spans are not additive GPU service time.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.stats import SparseRunStats
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from adapters.longlive_sparse.video_decode import decode_latents_chunked_exact


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=('rag_dense', 'transfer_vaware_hybrid_history'), required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('real GPU profiling required')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/'summary.json').exists():
        raise FileExistsError('completed profile is not rerun')
    os.environ.update(INFER_OUTPUT_DIR=str(root), LONGLIVE_CAPTURE_COMPLETE_ATTENTION='0', LONGLIVE_NVTX='1')
    prompt = next(c for c in json.loads((ROOT/'configs/system/profile_calibration_prompts.json').read_text())['candidates']
                  if c['prompt_id'] == 'calibration_motion')
    params = json.loads((ROOT/'configs/formal/method_params.json').read_text())['method_params'].get(args.method, {})
    config = yaml.safe_load((ROOT/'configs/inferhub/rag_method_21.yaml').read_text())
    (root/'empty_prompts.txt').write_text('')
    config.update(data_path=str(root/'empty_prompts.txt'), output_folder=str(root/'base_load'), inference_iter=0)
    config['sparse_history'].update(method=args.method, history_density=1. if args.method == 'rag_dense' else .25,
                                    method_params=params, record_per_call=True)
    system = LongLiveSystemConfig(profile_mode='trace', transfer_layout='exact_compact', staging_mode='persistent_separate',
        cpu_pack_policy='archive_runs', gpu_union_cache='per_chunk', gpu_union_cache_budget_mib=4096,
        archive_offload='pooled_pageable', host_pinned_budget_mib=128)
    config['longlive_system'] = system.as_dict()
    path = root/'load_config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    from scripts.run_longlive_sparse import run_config
    started = time.perf_counter()
    pipeline = run_config(path)['pipeline']
    load_s = time.perf_counter()-started
    state = {}
    def before(module, positional, keywords):
        if keywords.get('classify_mode', False):
            return
        start = int(keywords['current_start'])
        call = state['counts'].get(start, 0)
        state['counts'][start] = call+1
        active = start == state['trace_start'] and call == 0
        if active:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(f"optimized_{args.method}_lf{state['length']}_last_chunk_first_call")
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        state['current'] = (start, call, event, time.perf_counter(), active)
    def after(module, positional, keywords, output):
        if keywords.get('classify_mode', False):
            return
        start, call, begin, wall_start, active = state.pop('current')
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        state['calls'].append((start, call, begin, end, time.perf_counter()-wall_start))
        if active:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
            state['trace_completed'] = True
    hooks = [pipeline.generator.register_forward_pre_hook(before, with_kwargs=True),
             pipeline.generator.register_forward_hook(after, with_kwargs=True)]
    from utils.misc import set_seed
    records = []
    try:
        for length in (39, 120, 240):
            state.clear()
            state.update(length=length, trace_start=(length-3)*1560, counts={}, calls=[], trace_completed=False)
            pipeline.sparse_history_aggregate_stats = SparseRunStats(method=args.method)
            pipeline.sparse_history_completed_runs = []
            set_seed(20260904)
            noise = torch.randn(1, length, 16, 60, 104, device='cuda', dtype=torch.bfloat16)
            noise_sha = tensor_sha256(noise)
            started = time.perf_counter()
            with torch.inference_mode():
                _, latent = pipeline.inference(noise=noise, text_prompts=[prompt['prompt']], return_latents=True,
                                               low_memory=True, profile=True, skip_vae_decode=True)
            torch.cuda.synchronize()
            generation_s = time.perf_counter()-started
            if not state['trace_completed'] or not torch.isfinite(latent).all():
                raise RuntimeError('trace gate missing or latent invalid')
            call_rows = [{'current_start': start, 'pass': call, 'generator_gpu_span_s': begin.elapsed_time(end)/1000,
                           'generator_host_call_s': wall} for start, call, begin, end, wall in state['calls']]
            chunk_events = {}
            for start, call, begin, end, _ in state['calls']:
                chunk_events.setdefault(start, []).append((begin, end))
            chunk_spans = {start: events[0][0].elapsed_time(events[-1][1])/1000
                           for start, events in chunk_events.items()}
            steady = [value for start, value in chunk_spans.items()
                      if start >= 30*1560 and start != state['trace_start']]
            decode_started = time.perf_counter()
            with torch.inference_mode():
                video = decode_latents_chunked_exact(pipeline.vae, latent.cuda(), chunk_size=120)
            torch.cuda.synchronize()
            vae_s = time.perf_counter()-decode_started
            if video.shape[1] != 4*length-3 or not torch.isfinite(video).all():
                raise RuntimeError('continuous VAE output invalid')
            stats = pipeline.sparse_history_aggregate_stats.as_dict()
            stats['archive_storage'] = pipeline.sparse_history_archive.storage_summary()
            stats['history_union_cache'] = pipeline.history_union_cache.as_dict()
            stats_path = root/f'lf{length}_stats.json'
            stats_path.write_text(json.dumps(stats, indent=2)+'\n')
            torch.save(latent.cpu(), root/f'lf{length}_latents.pt')
            record = {'latent_frames': length, 'decoded_frames': int(video.shape[1]), 'initial_noise_sha256': noise_sha,
                'latent_sha256': tensor_sha256(latent), 'generation_diagnostic_s': generation_s,
                'vae_decode_s': vae_s, 'generator_calls': call_rows, 'profiled_latent': length-3,
                'steady_chunk_span_p50_s': statistics.median(steady),
                'steady_chunk_span_p95_s': sorted(steady)[min(len(steady)-1, round(.95*(len(steady)-1)))],
                'steady_unprofiled_chunk_samples': len(steady),
                'chunk_spans_first_generator_entry_to_last_exit_s': chunk_spans,
                'profiled_chunk_excluded_from_percentiles': True,
                'archive': stats['archive_storage'], 'component_service_times': stats['timing']}
            records.append(record)
            (root/f'lf{length}_record.json').write_text(json.dumps(record, indent=2)+'\n')
            print(json.dumps({k:v for k,v in record.items() if k not in ('generator_calls', 'component_service_times')}), flush=True)
            del video, latent, noise
    finally:
        for hook in hooks:
            hook.remove()
    (root/'summary.json').write_text(json.dumps({'status': 'pass', 'method': args.method, 'system': system.as_dict(),
        'model_load_s': load_s, 'records': records, 'gpu': torch.cuda.get_device_name(),
        'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'scope': 'optimized_real_39_120_240_generator_and_VAE_profile', 'capture_inside_trace': False,
        'trace_windows': 3, 'whole_video_speed_claim': False, 'encoder_included': False,
        'GPU_spans_include_launch_gaps_not_additive_service_times': True}, indent=2)+'\n')


if __name__ == '__main__':
    main()
