#!/usr/bin/env python3
"""Frozen development diagnostic: two-stage reuse and same-checkpoint pulses.

No formal prompts, no performance/Pareto claim. Pulse experiments compare
temporal retrieval policies, not semantic identity/state deletion categories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.checkpoint_probe import capture_checkpoint, replay_suffix
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.memory_dynamics import MemoryDynamicsObserver, frame_lifecycle
from adapters.longlive_sparse.offline_eval import output_error_metrics
from adapters.longlive_sparse.system_config import LongLiveSystemConfig


def prototype_sha(archive):
    sha = hashlib.sha256()
    for layer, frames in sorted(archive._layers.items()):
        for frame_id, frame in sorted(frames.items()):
            sha.update(f'{layer}:{frame_id}:'.encode())
            for value in (frame.block_centroids, frame.block_value_centroids):
                sha.update(tensor_sha256(value).encode())
    return sha.hexdigest()


@torch.inference_mode()
def run(args, root):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('real CUDA required')
    manifest = json.loads((ROOT / 'configs/system/memory_dynamics_probe.json').read_text())
    prompt = next(row for row in json.loads((ROOT / 'configs/system/profile_calibration_prompts.json').read_text())['candidates']
                  if row['prompt_id'] == args.prompt)
    source = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'], text=True).strip()
    if dirty and not args.allow_dirty_gate:
        raise RuntimeError('freeze source before a development batch')
    config = yaml.safe_load((ROOT / 'configs/inferhub/rag_method_21.yaml').read_text())
    params = json.loads((ROOT / 'configs/formal/method_params.json').read_text())['method_params']['transfer_vaware_hybrid_history']
    (root / 'empty_prompts.txt').write_text('')
    config.update(data_path=str(root/'empty_prompts.txt'), output_folder=str(root/'base_load'), inference_iter=0)
    config['sparse_history'].update(method='transfer_vaware_hybrid_history', history_density=.25,
        method_params=params, refresh_policy='per_chunk', record_per_call=True)
    system = LongLiveSystemConfig(transfer_layout='exact_compact', staging_mode='persistent_separate',
        cpu_pack_policy='archive_runs', gpu_union_cache='per_chunk', gpu_union_cache_budget_mib=768,
        archive_offload='pooled_pageable', host_pinned_budget_mib=128)
    config['longlive_system'] = system.as_dict()
    config_path = root / 'load_config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    os.environ['INFER_OUTPUT_DIR'] = str(root)
    os.environ['LONGLIVE_CAPTURE_CASE_TAG'] = args.prompt
    if not args.skip_complete_captures:
        os.environ.update(LONGLIVE_CAPTURE_COMPLETE_ATTENTION='1',
            LONGLIVE_COMPLETE_CAPTURE_LAYERS=','.join(map(str, manifest['complete_capture_layers'])),
            LONGLIVE_COMPLETE_CAPTURE_STARTS=','.join(str(x*1560) for x in manifest['complete_capture_latents'] if x < args.latent_frames),
            LONGLIVE_COMPLETE_CAPTURE_PASSES='5')
    from scripts.run_longlive_sparse import run_config
    started = time.perf_counter()
    pipeline = run_config(config_path)['pipeline']
    load_s = time.perf_counter() - started
    observer = MemoryDynamicsObserver(root/'route_observations', layers=manifest['route_layers'],
        shadow_starts=[x*1560 for x in manifest['shadow_latents'] if x < args.latent_frames])
    for module in pipeline.sparse_history_modules:
        module.memory_dynamics_observer = observer
    checkpoint_frame = manifest['pulse_start_latent']
    end_frame = checkpoint_frame + manifest['pulse_horizon_chunks'] * pipeline.num_frame_per_block
    gate = {'checkpoint': None, 'expected_future_prototype_sha': None}
    calls, seen = [], {}

    def before(module, positional, keywords):
        start = int(keywords['current_start'])
        index = seen.get(start, 0)
        seen[start] = index + 1
        observer.current_timestep = int(keywords['timestep'].reshape(-1)[0].item())
        if start == checkpoint_frame * 1560 and index == 0:
            gate['checkpoint'] = capture_checkpoint(pipeline, keywords)
        torch.cuda.synchronize()
        calls.append({'current_start': start, 'call_index': index,
                      'timestep': observer.current_timestep, 'started': time.perf_counter()})

    def after(module, positional, keywords, output):
        torch.cuda.synchronize()
        row = calls[-1]
        row['diagnostic_generator_wall_s'] = time.perf_counter() - row.pop('started')
        if row['timestep'] == 0:
            row['archive_storage'] = pipeline.sparse_history_archive.storage_summary()
            if row['current_start'] + 3*1560 == end_frame * 1560:
                gate['expected_future_prototype_sha'] = prototype_sha(pipeline.sparse_history_archive)

    hooks = [pipeline.generator.register_forward_pre_hook(before, with_kwargs=True),
             pipeline.generator.register_forward_hook(after, with_kwargs=True)]
    from utils.misc import set_seed
    set_seed(manifest['seed'])
    device = next(pipeline.generator.parameters()).device
    noise = torch.randn(1,args.latent_frames,16,60,104,device=device,dtype=torch.bfloat16)
    noise_sha = tensor_sha256(noise)
    started = time.perf_counter()
    _, latent = pipeline.inference(noise=noise, text_prompts=[prompt['prompt']], return_latents=True,
        low_memory=True, profile=False, skip_vae_decode=True)
    torch.cuda.synchronize()
    diagnostic_s = time.perf_counter() - started
    for hook in hooks:
        hook.remove()
    for module in pipeline.sparse_history_modules:
        module.memory_dynamics_observer = None
    os.environ.pop('LONGLIVE_CAPTURE_COMPLETE_ATTENTION', None)
    if not torch.isfinite(latent).all():
        raise RuntimeError('nonfinite uninterrupted trajectory')
    observer_sha = observer.finish()
    baseline_stats = pipeline.sparse_history_archive.stats.as_dict()
    (root/'sparse_history_stats.json').write_text(json.dumps(baseline_stats, indent=2)+'\n')
    (root/'generator_calls.json').write_text(json.dumps(calls, indent=2)+'\n')
    (root/'coarse_retrieval.json').write_text(json.dumps(pipeline.memory_indices_log, indent=2)+'\n')
    lifecycle = frame_lifecycle(pipeline.memory_indices_log, sink_size=1, recent_exclude=5, chunk_frames=3)
    (root/'coarse_lifecycle.json').write_text(json.dumps(lifecycle, indent=2)+'\n')
    torch.save(latent.cpu(), root/'uninterrupted_latents.pt')
    print(json.dumps({'stage':'uninterrupted_complete','diagnostic_s':diagnostic_s,
                      'route_observations':len(observer.records)}), flush=True)
    if gate['checkpoint'] is None or gate['expected_future_prototype_sha'] is None:
        raise RuntimeError('checkpoint or future commit audit not reached')
    expected = latent[:,checkpoint_frame:end_frame].cpu()
    results = []
    for policy in manifest['pulse_policies']:
        started = time.perf_counter()
        suffix, retrievals = replay_suffix(pipeline, gate['checkpoint'], noise,
            chunks=manifest['pulse_horizon_chunks'], policy=policy, pulse_seed=manifest['pulse_seed'])
        row = {'policy':policy, 'diagnostic_suffix_s':time.perf_counter()-started,
               'retrievals':retrievals, 'latent_sha256':tensor_sha256(suffix),
               'per_chunk': [output_error_metrics(expected[:,i:i+3], suffix[:,i:i+3])
                             for i in range(0, suffix.shape[1], 3)],
               'future_prototype_sha':prototype_sha(pipeline.sparse_history_archive),
               'same_coarse_frame_budget': True, 'semantic_role_claim': False}
        rows = pipeline.sparse_history_archive.stats.call_records
        pulse_rows = [r for r in rows if r['current_start'] == checkpoint_frame*1560]
        row['pulse_selected_tokens'] = sorted(set(r['selected_history_tokens'] for r in pulse_rows))
        row['pulse_candidate_tokens_per_head'] = sorted(set(r['candidate_history_tokens'] for r in pulse_rows))
        if policy == 'reference':
            row['bitwise_latent_match'] = torch.equal(expected, suffix)
            row['future_prototype_match'] = row['future_prototype_sha'] == gate['expected_future_prototype_sha']
            expected_routes = [r['route_plan_sha256'] for r in baseline_stats['call_records']
                               if checkpoint_frame*1560 <= r['current_start'] < end_frame*1560]
            row['ordered_route_match'] = expected_routes == [r['route_plan_sha256'] for r in rows]
            if not all(row[key] for key in ('bitwise_latent_match', 'future_prototype_match', 'ordered_route_match')):
                (root/'failed_noop_gate.json').write_text(json.dumps(row,indent=2)+'\n')
                raise RuntimeError('no-op suffix/future-state reproduction failed; pulses prohibited')
        else:
            if (row['pulse_selected_tokens'] != results[0]['pulse_selected_tokens'] or
                row['pulse_candidate_tokens_per_head'] != results[0]['pulse_candidate_tokens_per_head']):
                raise RuntimeError('pulse does not match actual Final token budget')
        torch.save(suffix, root/f'pulse_{policy}_latents.pt')
        (root/f'pulse_{policy}.json').write_text(json.dumps(row,indent=2)+'\n')
        results.append(row)
        print(json.dumps({'stage':'pulse_complete', **row}), flush=True)
    return {'status':'pass','scope':'development_memory_diagnostic_not_quality_or_speed_trial',
            'prompt':prompt,'seed':manifest['seed'],'latent_frames':args.latent_frames,
            'source_commit':source,'source_dirty_gate_only':bool(dirty),
            'manifest_sha256':hashlib.sha256((ROOT/'configs/system/memory_dynamics_probe.json').read_bytes()).hexdigest(),
            'initial_noise_sha256':noise_sha,'uninterrupted_latent_sha256':tensor_sha256(latent),
            'observation_sha256':observer_sha,'gpu':torch.cuda.get_device_name(),
            'model_load_s':load_s,'instrumented_generation_s':diagnostic_s,
            'system':system.as_dict(),'route_observation_count':len(observer.records),
            'pulses':results, 'formal_holdouts_used':False, 'vae_decoded':False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', choices=['calibration_motion','calibration_state'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--latent-frames', type=int, choices=[39,120,240], default=120)
    parser.add_argument('--allow-dirty-gate', action='store_true')
    parser.add_argument('--skip-complete-captures', action='store_true')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # The runner may have made the directory for logs; a claim forbids repeats.
    with (root/'probe_claim.json').open('x') as handle:
        json.dump(vars(args), handle, indent=2)
    try:
        result = run(args, root)
    except Exception as error:
        (root/'terminal.json').write_text(json.dumps({'status':'fail','error_type':type(error).__name__,
                                                     'error':str(error)},indent=2)+'\n')
        raise
    (root/'terminal.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
