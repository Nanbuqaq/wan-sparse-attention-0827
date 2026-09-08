#!/usr/bin/env python3
"""One-pass causal prototype-tail EFFECT reference, with all added costs exposed.

This deliberately constructs moments from fully materialized committed history
after raw selection. It is not an optimized sparse onload/Pareto candidate.
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

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.prototype_reference_runtime import PrototypeReferenceRuntime
from adapters.longlive_sparse.committed_moment_runtime import CommittedMomentRuntime
from adapters.longlive_sparse.precision_runtime import PrecisionRecipe,WholeBlockPrecisionRuntime
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.stream_video_sink import IncrementalVideoSink
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from adapters.longlive_sparse.video_decode import decode_latents_chunked_exact
from adapters.longlive_sparse.offline_eval import output_error_metrics


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--kind', choices=('motion', 'state'), required=True)
    p.add_argument('--latent-frames', type=int, default=39)
    p.add_argument('--reverse-tail-order', action='store_true')
    p.add_argument('--experiment', choices=('reference', 'committed_moments','precision_wire'), default='reference')
    p.add_argument('--seed',type=int,default=20260904)
    p.add_argument('--variants',help='Explicit non-duplicated subset; legacy_final must remain the first control')
    p.add_argument('--local-rope-layout',choices=('upstream','direct_output'),default='upstream')
    p.add_argument('--exact-local-window-frames',type=int,
                   help='Method ablation, including current chunk; applied equally to all selected arms')
    args = p.parse_args()
    if args.latent_frames < 21 or args.latent_frames % 3:
        p.error('complete block-aligned history trajectory required')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('real CUDA required')
    os.environ.update(INFER_OUTPUT_DIR=str(args.output), LONGLIVE_CAPTURE_QKV='0', LONGLIVE_CAPTURE_COMPLETE_ATTENTION='0', LONGLIVE_NVTX='0')
    prompt = next(p for p in json.loads((ROOT/'configs/system/profile_calibration_prompts.json').read_text())['candidates']
                  if p['prompt_id'] == 'calibration_'+args.kind)
    params = json.loads((ROOT/'configs/formal/method_params.json').read_text())['method_params']['transfer_vaware_hybrid_history']
    system = LongLiveSystemConfig(transfer_layout='exact_compact', staging_mode='persistent_separate',
        cpu_pack_policy='archive_runs', gpu_union_cache='per_chunk', gpu_union_cache_budget_mib=4096,
        archive_offload='pooled_pageable', host_pinned_budget_mib=128, route_metadata_mode='validated_reuse',
        execution_dataflow='qout_resident_grouped_fa2')
    system=replace(system,local_rope_layout=args.local_rope_layout)
    config = yaml.safe_load((ROOT/'configs/inferhub/rag_method_21.yaml').read_text())
    (args.output/'empty_prompts.txt').write_text('')
    config.update(data_path=str(args.output/'empty_prompts.txt'), output_folder=str(args.output/'load'), inference_iter=0)
    config['sparse_history'].update(method='transfer_vaware_hybrid_history', backend='resident_grouped_fa2', history_density=.25,
                                    refresh_policy='per_chunk', method_params=params, record_per_call=True)
    config['longlive_system'] = system.as_dict()
    config_path = args.output/'config.yaml'; config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    from scripts.run_longlive_sparse import run_config
    from adapters.longlive_sparse.runtime import configure_pipeline_system
    start = time.perf_counter(); pipeline = run_config(config_path)['pipeline']; load_s = time.perf_counter()-start
    from utils.misc import set_seed
    variants = [('legacy_final', None, 64), ('sdpa_null', 'sdpa_null', 64),
                ('prototype_tail64', 'prototype_tail', 64), ('prototype_tail16', 'prototype_tail', 16)]
    if args.experiment == 'committed_moments':
        variants = [('legacy_final',None,64),('sdpa_null','sdpa_null',64),
                    ('prototype_tail16','prototype_tail',16),
                    ('committed_tail16','committed_spatial',16),
                    ('committed_key4','committed_key_kmeans',64)]
    if args.experiment=='precision_wire':
        variants=[('legacy_final',None,64),('sdpa_null','sdpa_null',64),
                  ('wire_random14','wire_random',64),('wire_mass14','wire_mass_value',64),
                  ('wire_risk14','wire_mass_key_variance',64)]
    if args.variants:
        names=args.variants.split(',');available={row[0]:row for row in variants}
        if not names or names[0]!='legacy_final' or len(names)!=len(set(names)) or any(name not in available for name in names):
            raise ValueError('invalid variant subset or missing first legacy control')
        variants=[available[name] for name in names]
    if args.reverse_tail_order:
        variants[-2:] = list(reversed(variants[-2:]))
    report = dict(status='running', prompt=prompt, seed=args.seed, latent_frames=args.latent_frames,
        gpu=torch.cuda.get_device_name(), model_load_s=load_s, source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [Path(__file__), *sorted((ROOT/'adapters/longlive_sparse').glob('*.py'))]},
        variants=[], online_Pareto_eligible=False, two_pass_generation=False,
        experiment=args.experiment,
        interpretation='causal effect comparison; distinguish full-candidate reference from commit-time summaries; all overhead charged')
    reference_latent = None
    base_sparse_config=pipeline.sparse_history_config
    for name, mode, block in variants:
        root = args.output/name; root.mkdir()
        for module in pipeline.sparse_history_modules:
            module.clear_selection_cache()
        if mode and mode.startswith('wire_'):
            current_sparse=replace(base_sparse_config,method='whole_block_precision_history',history_density=.14,
                method_params={'precision_query_samples':1024,'precision_variance_codec':'u8_scaled',
                               'precision_admission':mode.removeprefix('wire_')})
        else:current_sparse=base_sparse_config
        if args.exact_local_window_frames is not None:
            current_sparse=replace(current_sparse,method_params=dict(current_sparse.method_params,
                exact_local_window_frames=args.exact_local_window_frames))
        pipeline.sparse_history_config=current_sparse
        pipeline.sparse_history_archive.config=current_sparse
        for module in pipeline.sparse_history_modules:module.sparse_config=current_sparse
        configure_pipeline_system(pipeline, system)
        set_seed(args.seed)
        noise = torch.randn(1,args.latent_frames,16,60,104,device='cuda',dtype=torch.bfloat16)
        if mode and mode.startswith('wire_'):
            wrapper=WholeBlockPrecisionRuntime(pipeline,PrecisionRecipe.from_config(current_sparse))
        elif mode and mode.startswith('committed_'):
            wrapper = CommittedMomentRuntime(pipeline,block_tokens=block,grouping=mode.removeprefix('committed_'))
        else:
            wrapper = PrototypeReferenceRuntime(pipeline, mode=mode, block_tokens=block) if mode else None
        started = time.perf_counter()
        try:
            with wrapper if wrapper is not None else contextlib.nullcontext():
                _, latent = pipeline.inference(noise=noise,text_prompts=[prompt['prompt']],return_latents=True,
                    low_memory=True,profile=False,skip_vae_decode=True)
            torch.cuda.synchronize()
            generation_s = time.perf_counter()-started
            raw = decode_latents_chunked_exact(pipeline.vae, latent, chunk_size=120)
            sink = IncrementalVideoSink(root/'video.mp4',expected_frames=4*args.latent_frames-3,started=started)
            sink(raw); sink_metrics = sink.close()
            if not bool(torch.isfinite(latent).all()):
                raise RuntimeError('nonfinite prototype-reference trajectory')
            current_latent = latent.cpu()
            if reference_latent is None:
                reference_latent = current_latent.clone()
            stats = pipeline.sparse_history_archive.stats.as_dict()
            extra = wrapper.audit() if wrapper is not None else dict(extra_candidate_H2D_bytes=0,extra_metadata_H2D_bytes=0,records=[])
            history_all_H2D=stats['transferred_bytes']+stats['restore_index_h2d_bytes']+stats['rope_metadata_h2d_bytes']+extra['extra_candidate_H2D_bytes']+extra['extra_metadata_H2D_bytes']
            first_route_candidate_bytes=(extra.get('ledger',{}).get('candidate_KV_bytes_at_route_miss') or
                sum(r['candidate_transfer_bytes'] for r in stats['call_records'] if r['denoising_pass']==0))
            full_H2D_fraction=history_all_H2D/first_route_candidate_bytes if first_route_candidate_bytes else 0.
            if mode and mode.startswith('wire_') and full_H2D_fraction>.25:
                raise RuntimeError(f'precision wire exceeds full history H2D budget: {full_H2D_fraction}')
            torch.save(current_latent,root/'latents.pt')
            record = dict(variant=name,status='pass',effective_method=name,
                method_registry_id=current_sparse.method,method_config=current_sparse.as_dict(),
                base_raw_selector=mode.removeprefix('wire_') if mode and mode.startswith('wire_') else 'legacy_Final_70_15_15',
                prototype_block_tokens=block if mode and mode != 'sdpa_null' else None,
                initial_noise_sha256=tensor_sha256(noise),latent_sha256=tensor_sha256(current_latent),
                raw_RGB_sha256=sink_metrics['raw_RGB_sha256'],video=str(root/'video.mp4'),
                generation_wall_s=generation_s,complete_wall_s=time.perf_counter()-started,
                latent_error_vs_legacy_not_absolute_quality=output_error_metrics(reference_latent,current_latent),
                base_runtime_reported_H2D_bytes=stats['transferred_bytes'],
                all_history_H2D_bytes_including_wire_and_restore_RoPE_metadata=history_all_H2D,
                all_history_H2D_over_first_route_candidates=full_H2D_fraction,
                additional_reference_H2D_bytes=extra['extra_candidate_H2D_bytes']+extra['extra_metadata_H2D_bytes'],
                prototype_reference=extra,only_completed_history_used=True,optimized_onload_claim=False,
                full_candidate_tail_materialization=mode=='prototype_tail',
                additional_index_D2H_bytes=extra.get('extra_index_D2H_bytes',0),
                mechanism=('causal_whole_block_precision' if mode and mode.startswith('wire_') else
                           'commit_time_summary_onload' if mode and mode.startswith('committed_') else 'effect_reference'))
            (root/'stats.json').write_text(json.dumps(stats,indent=2)+'\n')
            (root/'terminal.json').write_text(json.dumps(record,indent=2)+'\n')
            report['variants'].append(record)
            print(json.dumps({k:v for k,v in record.items() if k != 'prototype_reference'}),flush=True)
            del latent,raw,current_latent,noise,wrapper
        except BaseException:
            failure=dict(variant=name,status='fail',traceback=traceback.format_exc())
            (root/'terminal.json').write_text(json.dumps(failure,indent=2)+'\n')
            report['variants'].append(failure)
            report['status']='fail'; (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
            raise
        finally:
            pipeline.vae.model.clear_cache(); torch.cuda.empty_cache(); gc.collect()
        (args.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    report['status']='pass'
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
