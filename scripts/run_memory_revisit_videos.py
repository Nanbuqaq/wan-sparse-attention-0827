#!/usr/bin/env python3
"""Dense-only feasibility screen for declared streaming prompt events."""
import argparse
import contextlib
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.scheduled_prompts import ScheduledPromptContext
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.stream_video_sink import IncrementalVideoSink
from adapters.longlive_sparse.video_decode import decode_latents_chunked_exact
from adapters.longlive_sparse.offline_eval import output_error_metrics


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenario',type=int,choices=(0,1),required=True)
    p.add_argument('--seed',type=int,default=20260904)
    p.add_argument('--mode',choices=('null_gate','dense_screen'),default='dense_screen')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('real CUDA required')
    spec_path=ROOT/'configs/system/memory_revisit_development.json';spec=json.loads(spec_path.read_text())
    scenario=spec['scenarios'][args.scenario]
    if args.mode=='dense_screen' and args.seed not in spec['seeds']:
        raise ValueError('Dense screening seeds must match the predeclared development configuration')
    length=21 if args.mode=='null_gate' else spec['latent_frames']
    segments=scenario['segments']
    if args.mode=='null_gate':
        same=[dict(s,start_latent=i*6,prompt=segments[0]['prompt']) for i,s in enumerate(segments)]
        short=[dict(s,start_latent=i*6) for i,s in enumerate(segments)]
        variants=[('single_prompt_reference',None),('same_prompt_events',same),('switch_branch_smoke',short)]
    else:variants=[('scheduled_Dense',segments)]
    os.environ.update(INFER_OUTPUT_DIR=str(args.output),LONGLIVE_CAPTURE_QKV='0',LONGLIVE_CAPTURE_COMPLETE_ATTENTION='0',LONGLIVE_NVTX='0')
    system=LongLiveSystemConfig(transfer_layout='exact_compact',staging_mode='persistent_separate',cpu_pack_policy='archive_runs',
        gpu_union_cache='per_chunk',gpu_union_cache_budget_mib=4096,archive_offload='pooled_pageable',
        host_pinned_budget_mib=128,route_metadata_mode='validated_reuse',execution_dataflow='qout_resident_grouped_fa2',
        local_rope_layout='direct_output')
    config=yaml.safe_load((ROOT/'configs/inferhub/rag_method_21.yaml').read_text())
    (args.output/'empty_prompts.txt').write_text('')
    config.update(data_path=str(args.output/'empty_prompts.txt'),output_folder=str(args.output/'load'),inference_iter=0)
    config['sparse_history'].update(method='rag_dense',backend='resident_grouped_fa2',history_density=1.,refresh_policy='per_chunk',record_per_call=True,method_params={})
    config['longlive_system']=system.as_dict()
    path=args.output/'config.yaml';path.write_text(yaml.safe_dump(config,sort_keys=False))
    from scripts.run_longlive_sparse import run_config
    from adapters.longlive_sparse.runtime import configure_pipeline_system
    from utils.misc import set_seed
    begin=time.perf_counter();pipeline=run_config(path)['pipeline'];load_s=time.perf_counter()-begin
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    report=dict(status='running',mode=args.mode,scenario=scenario,seed=args.seed,latent_frames=length,
        source_commit=commit,spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),gpu=torch.cuda.get_device_name(),
        torch_version=str(torch.__version__),cuda_version=torch.version.cuda,model_load_s=load_s,variants=[],
        automatic_semantic_validity=False,formal_holdout=False,quality_review_pending=True,
        future_generated_output_access=False,upstream_interactive_reproduction=False)
    reference=None;reference_identity=None
    for name,event_segments in variants:
        root=args.output/name;root.mkdir()
        for module in pipeline.sparse_history_modules:module.clear_selection_cache()
        configure_pipeline_system(pipeline,system)
        schedule=ScheduledPromptContext(pipeline,event_segments,latent_frames=length) if event_segments is not None else None
        start=time.perf_counter()
        try:
            with schedule if schedule is not None else contextlib.nullcontext():
                set_seed(args.seed)
                noise=torch.randn(1,length,16,60,104,device='cuda',dtype=torch.bfloat16)
                _,latent=pipeline.inference(noise=noise,text_prompts=[segments[0]['prompt']],return_latents=True,
                    low_memory=True,profile=False,skip_vae_decode=True)
            torch.cuda.synchronize();generation_s=time.perf_counter()-start
            if not torch.isfinite(latent).all():raise RuntimeError('nonfinite trajectory')
            raw=decode_latents_chunked_exact(pipeline.vae,latent,chunk_size=120)
            sink=IncrementalVideoSink(root/'video.mp4',expected_frames=4*length-3,started=start)
            sink(raw);pixels=sink.close()
            sampling_s=time.perf_counter()-start
            current_latent=latent.detach().cpu()
            # Persist a successful generation before fallible post-hoc audit.
            torch.save(current_latent,root/'latents.pt')
            stats=pipeline.sparse_history_archive.stats.as_dict()
            ordered=[(r['layer_id'],r['current_start'],r['denoising_pass'],r['route_plan_sha256']) for r in stats['call_records']]
            identity=dict(noise=tensor_sha256(noise),latent=tensor_sha256(current_latent),raw_RGB=pixels['raw_RGB_sha256'],
                ordered_routes=hashlib.sha256(json.dumps(ordered).encode()).hexdigest())
            if reference is None:reference=current_latent.clone();reference_identity=identity
            if name=='same_prompt_events' and identity!=reference_identity:
                raise RuntimeError('noop prompt events changed the whole trajectory')
            schedule_audit=schedule.audit() if schedule is not None else None
            if schedule_audit is not None and len(schedule_audit['events'])!=len(event_segments):
                raise RuntimeError('not all declared event branches executed')
            case_key=dict(commit=commit,spec_sha256=report['spec_sha256'],scenario=scenario['id'],seed=args.seed,
                latent_frames=length,variant=name,segments=event_segments,system=system.as_dict())
            record=dict(variant=name,status='pass',identity=identity,case_key=case_key,
                case_identity_sha256=hashlib.sha256(json.dumps(case_key,sort_keys=True).encode()).hexdigest(),
                generation_including_preencode_s=generation_s,complete_wall_s=time.perf_counter()-start,
                generation_decode_encode_s=sampling_s,
                schedule=schedule_audit,video=str(root/'video.mp4'),history_H2D_bytes=stats['transferred_bytes'],
                nominal_history_density=1.,actual_fine_method='rag_dense',
                fidelity_to_single_prompt_not_a_task_score=output_error_metrics(reference,current_latent),
                semantic_validity='pending_manual_review',technical_pass_is_not_task_success=True)
            (root/'stats.json').write_text(json.dumps(stats,indent=2)+'\n')
            (root/'retrieval.json').write_text(json.dumps(pipeline.memory_indices_log,indent=2)+'\n')
            (root/'terminal.json').write_text(json.dumps(record,indent=2)+'\n')
            report['variants'].append(record)
            print(json.dumps({k:v for k,v in record.items() if k not in ('schedule','case_key')}),flush=True)
            del noise,latent,raw,current_latent
        except BaseException:
            failure=dict(variant=name,status='fail',traceback=traceback.format_exc())
            report['variants'].append(failure);report['status']='fail'
            (root/'terminal.json').write_text(json.dumps(failure,indent=2)+'\n')
            (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
            raise
        finally:pipeline.vae.model.clear_cache();torch.cuda.empty_cache();gc.collect()
        (args.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    report['status']='pass';(args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
