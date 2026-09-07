#!/usr/bin/env python3
"""Real generation -> completed-latent VAE stream -> incremental MP4 factorial.

One loaded model, same seed and attention route. Explicit diagnostic controls
separate VAE scheduling from device/current-stream timing fences. No history
prefetch or prototype-tail method is implicitly enabled.
"""
import argparse
import contextlib
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import av
import torch
import random
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.streaming_vae import StreamingVAEDecoder
from adapters.longlive_sparse.stream_video_sink import IncrementalVideoSink
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from adapters.longlive_sparse.video_decode import decode_latents_chunked_exact


def build_repetition_schedule(variants, repeats, seed):
    if repeats < 1 or not variants or len(set(variants)) != len(variants):
        raise ValueError('distinct nonempty arms and positive repetitions required')
    rng = random.Random(seed)
    schedule = []
    for repetition in range(repeats):
        order = list(variants)
        if repeats > 1:
            rng.shuffle(order)
        schedule.extend((repetition, *variant) for variant in order)
    return schedule


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--method', choices=('rag_dense', 'transfer_vaware_hybrid_history'), required=True)
    p.add_argument('--prompt', choices=('calibration_motion', 'calibration_state'), required=True)
    p.add_argument('--latent-frames', type=int, default=39)
    p.add_argument('--reverse-order', action='store_true')
    variant_names = ('batch_device', 'batch_current_stream', 'async_device', 'async_current_stream', 'async_priority_current_stream')
    p.add_argument('--profile-variant', choices=variant_names)
    p.add_argument('--only-variant', choices=variant_names)
    p.add_argument('--include-priority', action='store_true')
    p.add_argument('--variants', help='Explicit comma-separated subset of declared variants')
    p.add_argument('--warmup-latents', type=int, default=0)
    p.add_argument('--reference-summary', type=Path)
    p.add_argument('--repeats', type=int, default=1,
                   help='Fresh complete trajectories per arm; randomized blocked order after optional warmup')
    p.add_argument('--order-seed', type=int, default=20260908)
    p.add_argument('--rope-factorial', action='store_true',
                   help='Cross each selected pipeline arm with upstream/direct-output dense RoPE')
    args = p.parse_args()
    if args.repeats < 1 or (args.repeats > 1 and args.profile_variant):
        p.error('positive repetitions required; representative profiling must be a separate single repetition')
    if args.rope_factorial and (args.profile_variant or args.reference_summary):
        p.error('factorial uses its own complete equivalence reference; profile separately')
    if args.latent_frames % 3 or args.latent_frames < 21:
        p.error('block-aligned history trajectory required')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('real CUDA required')
    os.environ.update(INFER_OUTPUT_DIR=str(args.output), LONGLIVE_CAPTURE_QKV='0', LONGLIVE_CAPTURE_COMPLETE_ATTENTION='0', LONGLIVE_NVTX='0')
    prompt = next(x for x in json.loads((ROOT/'configs/system/profile_calibration_prompts.json').read_text())['candidates'] if x['prompt_id'] == args.prompt)
    params = json.loads((ROOT/'configs/formal/method_params.json').read_text())['method_params'].get(args.method, {})
    config = yaml.safe_load((ROOT/'configs/inferhub/rag_method_21.yaml').read_text())
    (args.output/'empty_prompts.txt').write_text('')
    config.update(data_path=str(args.output/'empty_prompts.txt'), output_folder=str(args.output/'load'), inference_iter=0)
    config['sparse_history'].update(method=args.method, method_params=params, backend='resident_grouped_fa2',
        history_density=1. if args.method == 'rag_dense' else .25, refresh_policy='per_chunk', record_per_call=True)
    system = LongLiveSystemConfig(transfer_layout='exact_compact', staging_mode='persistent_separate',
        cpu_pack_policy='archive_runs', gpu_union_cache='per_chunk', gpu_union_cache_budget_mib=4096,
        archive_offload='pooled_pageable', host_pinned_budget_mib=128, route_metadata_mode='validated_reuse',
        execution_dataflow='qout_resident_grouped_fa2')
    config['longlive_system'] = system.as_dict()
    config_path = args.output/'load_config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    from scripts.run_longlive_sparse import run_config
    from adapters.longlive_sparse.runtime import configure_pipeline_system
    started = time.perf_counter()
    pipeline = run_config(config_path)['pipeline']
    from utils.misc import set_seed
    load_s = time.perf_counter()-started
    variants = [('batch', 'device'), ('batch', 'current_stream'), ('async', 'device'), ('async', 'current_stream')]
    if args.include_priority or args.only_variant == 'async_priority_current_stream':
        variants.append(('async_priority', 'current_stream'))
    if args.reverse_order:
        variants.reverse()
    if args.only_variant:
        variants = [(m, s) for m, s in variants if m+'_'+s == args.only_variant]
    if args.variants:
        requested = args.variants.split(',')
        if len(set(requested)) != len(requested) or any(x not in variant_names for x in requested):
            raise ValueError('invalid or duplicate variant list')
        all_variants = [('batch', 'device'), ('batch', 'current_stream'), ('async', 'device'),
                        ('async', 'current_stream'), ('async_priority', 'current_stream')]
        by_name = {m+'_'+s: (m, s) for m, s in all_variants}
        variants = [by_name[name] for name in requested]
        if args.reverse_order:
            variants.reverse()
    report = dict(status='running', method=args.method, prompt=prompt, latent_frames=args.latent_frames, seed=20260904,
        gpu=torch.cuda.get_device_name(), model_load_s=load_s, profile_in_upstream=False,
        source_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), *sorted((ROOT/'adapters/longlive_sparse').glob('*.py'))]},
        variants=[], no_history_onload_or_offload_overlap_enabled=True, independent_timing_repeats=False,
        complete_trajectory_repetitions=args.repeats, repetition_scope='same_loaded_process_fresh_archive_noise_and_outputs',
        order_seed=args.order_seed, model_load_excluded_from_samples=True,
        untimed_model_warmup_requested=bool(args.warmup_latents))
    if args.warmup_latents:
        if args.warmup_latents < 21 or args.warmup_latents % 3:
            raise ValueError('warmup must exercise history with block-aligned length')
        set_seed(20260904)
        device = next(pipeline.generator.parameters()).device
        warm_noise = torch.randn(1, args.warmup_latents, 16, 60, 104, dtype=torch.bfloat16, device=device)
        warm_started = time.perf_counter()
        warm_video, warm_latent = pipeline.inference(noise=warm_noise, text_prompts=[prompt['prompt']],
            return_latents=True, low_memory=True, profile=False, skip_vae_decode=False)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(warm_latent).all() and torch.isfinite(warm_video).all()):
            raise RuntimeError('warmup produced nonfinite values')
        report['untimed_warmup'] = dict(latents=args.warmup_latents, wall_s=time.perf_counter()-warm_started)
        del warm_noise, warm_video, warm_latent
        pipeline.vae.model.clear_cache()
        gc.collect()
    reference = None
    if args.reference_summary:
        external = json.loads(args.reference_summary.read_text())
        if (external['method'] != args.method or external['prompt']['prompt_id'] != args.prompt or
                external['prompt']['prompt'] != prompt['prompt'] or external['seed'] != 20260904 or
                external['latent_frames'] != args.latent_frames or external['status'] != 'pass'):
            raise ValueError('external identity reference must match the complete successful trajectory')
        reference = external['variants'][0]['identity']
        report['external_reference_summary_sha256'] = hashlib.sha256(args.reference_summary.read_bytes()).hexdigest()
    experiments = [(m,s,layout) for m,s in variants
                   for layout in (('upstream','direct_output') if args.rope_factorial else ('upstream',))]
    schedule = build_repetition_schedule(experiments, args.repeats, args.order_seed)
    report['execution_schedule'] = [dict(repetition=i,variant=m+'_'+s,local_rope_layout=layout) for i,m,s,layout in schedule]
    reference_name = external['variants'][0]['variant'] if args.reference_summary else None
    for repetition, mode, sync, rope_layout in schedule:
        name = mode+'_'+sync+(f'__rope_{rope_layout}' if args.rope_factorial else '')
        root = args.output/(name if args.repeats == 1 else name+f'__rep{repetition:02d}')
        root.mkdir()
        selected_system = replace(system, cuda_sync_scope=sync, local_rope_layout=rope_layout)
        for module in pipeline.sparse_history_modules:
            module.clear_selection_cache()
        configure_pipeline_system(pipeline, selected_system)
        set_seed(20260904)
        device = next(pipeline.generator.parameters()).device
        noise = torch.randn(1, args.latent_frames, 16, 60, 104, dtype=torch.bfloat16, device=device)
        noise_sha = tensor_sha256(noise)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        generator_stream = torch.cuda.Stream(device=device, priority=-1) if mode == 'async_priority' else None
        begin = time.perf_counter()
        sink = IncrementalVideoSink(root/'video.mp4', expected_frames=4*args.latent_frames-3, started=begin)
        decoder = StreamingVAEDecoder(pipeline.vae, device=device, dtype=noise.dtype, collect=False, on_chunk=sink) if mode.startswith('async') else None
        phase_counts = {}
        def completed(module, values, kwargs):
            start = int(kwargs['current_start'])
            phase = phase_counts.get(start, 0)
            phase_counts[start] = phase+1
            if phase == len(pipeline.denoising_step_list):
                if int(kwargs['timestep'].reshape(-1)[0]) != 0:
                    raise RuntimeError('completed-latent hook must be the clean commit call')
                if decoder is not None:
                    decoder.submit(kwargs['noisy_image_or_video'], start_latent=start//pipeline.frame_seq_length)
        hook = pipeline.generator.register_forward_pre_hook(completed, with_kwargs=True)
        profiling = args.profile_variant == name
        if profiling:
            os.environ['LONGLIVE_NVTX'] = '1'
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push('dataflow_full_pipeline_'+name)
        try:
            if profiling:
                torch.cuda.nvtx.range_push('streaming_pipeline/generation')
            generator_scope = torch.cuda.stream(generator_stream) if generator_stream is not None else contextlib.nullcontext()
            with generator_scope:
                _, latent = pipeline.inference(noise=noise, text_prompts=[prompt['prompt']], return_latents=True,
                    low_memory=True, profile=False, skip_vae_decode=True)
                torch.cuda.current_stream(device).synchronize()
            generation_s = time.perf_counter()-begin
            if profiling:
                torch.cuda.nvtx.range_pop()
            hook.remove()
            if decoder is not None:
                _, decode_metrics = decoder.finish()
            else:
                raw = decode_latents_chunked_exact(pipeline.vae, latent, chunk_size=120)
                sink(raw)
                del raw
                decode_metrics = None
            sink_metrics = sink.close()
            pipeline_s = time.perf_counter()-begin
            if profiling:
                torch.cuda.nvtx.range_pop()
                torch.cuda.cudart().cudaProfilerStop()
                os.environ['LONGLIVE_NVTX'] = '0'
            stats = pipeline.sparse_history_archive.stats.as_dict()
            ordered = [(r['layer_id'], r['current_start'], r['denoising_pass'], r['route_plan_sha256']) for r in stats['call_records']]
            identity = dict(noise=noise_sha, latent=tensor_sha256(latent), raw_RGB=sink_metrics['raw_RGB_sha256'],
                            ordered_routes=hashlib.sha256(json.dumps(ordered).encode()).hexdigest())
            if reference is None:
                reference = identity
                reference_name = name
            equivalent = identity == reference
            torch.save(latent.cpu(), root/'latents.pt')
            with av.open(str(root/'video.mp4')) as container:
                decoded_frames = sum(1 for _ in container.decode(video=0))
            if decoded_frames != 4*args.latent_frames-3:
                raise RuntimeError('encoded stream frame count mismatch')
            record = dict(variant=name, repetition=repetition, status='pass' if equivalent else 'negative', identity=identity,
                exact_reference=equivalent, reference_variant=reference_name,
                generator_stream_priority=generator_stream.priority if generator_stream is not None else 0,
                VAE_stream_priority=decoder.stream.priority if decoder is not None else None,
                generation_wall_including_hook_s=generation_s, generation_decode_encode_s=pipeline_s,
                complete_wall_including_audit_s=time.perf_counter()-begin, sink=sink_metrics,
                streaming_decoder=decode_metrics, system=selected_system.as_dict(),
                peak_GPU_bytes=torch.cuda.max_memory_allocated(),
                archive_storage=pipeline.sparse_history_archive.storage_summary(),
                video=str(root/'video.mp4'), decoded_frames=decoded_frames,
                diagnostic_timing=profiling, first_client_display_time_measured=False)
            (root/'stats.json').write_text(json.dumps(stats, indent=2)+'\n')
            (root/'terminal.json').write_text(json.dumps(record, indent=2)+'\n')
            report['variants'].append(record)
            print(json.dumps({k: v for k, v in record.items() if k not in ('streaming_decoder', 'system', 'archive_storage')}), flush=True)
        except BaseException:
            hook.remove()
            record = dict(variant=name, repetition=repetition, status='fail', traceback=traceback.format_exc())
            (root/'terminal.json').write_text(json.dumps(record, indent=2)+'\n')
            report['variants'].append(record)
            report['status'] = 'fail'
            (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
            raise
        finally:
            if decoder is not None and not decoder.closed:
                try:
                    decoder.finish()
                except Exception as cleanup_error:
                    print('VAE_CLEANUP_ERROR', repr(cleanup_error), flush=True)
            pipeline.vae.model.clear_cache()
            gc.collect()
        (args.output/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
    report['status'] = 'pass' if all(x['status'] == 'pass' for x in report['variants']) else 'negative'
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
