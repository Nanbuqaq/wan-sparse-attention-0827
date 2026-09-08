#!/usr/bin/env python3
"""Use the locked native interactive pipeline to audit prompt-switch recaching.

Shares the loaded backbone/LoRA and tested exact local attention adapter.
No historical archive or onload. The upstream interactive implementation is
read-only and its source hash is recorded. Cross-only is an explicit control.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.stream_video_sink import IncrementalVideoSink


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260909);p.add_argument('--gate',action='store_true')
    p.add_argument('--novel-control',choices=('duck','empty'));args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('real CUDA required')
    os.environ.update(INFER_OUTPUT_DIR=str(args.output),LONGLIVE_CAPTURE_QKV='0',LONGLIVE_CAPTURE_COMPLETE_ATTENTION='0',LONGLIVE_NVTX='0')
    spec=json.loads((ROOT/'configs/system/memory_revisit_development.json').read_text())['scenarios'][1]
    segments=[dict(s) for s in spec['segments']]
    if args.novel_control:
        controls=json.loads((ROOT/'configs/system/event_retrieval_negative_controls.json').read_text())
        segments[-1]['prompt']=controls['controls'][args.novel_control]
    length=39 if args.gate else 120
    if args.gate:segments=[dict(s,start_latent=t) for s,t in zip(segments,(0,6,12,30))]
    config=yaml.safe_load((ROOT/'configs/inferhub/rag_method_21.yaml').read_text())
    (args.output/'empty_prompts.txt').write_text('')
    config.update(data_path=str(args.output/'empty_prompts.txt'),output_folder=str(args.output/'load'),inference_iter=0)
    config['model_kwargs']['memory_size']=0
    config['sparse_history'].update(method='native_block',backend='resident_grouped_fa2',history_density=1.,refresh_policy='per_chunk',method_params={})
    config['longlive_system']=LongLiveSystemConfig(local_rope_layout='direct_output',execution_dataflow='qout_resident_grouped_fa2').as_dict()
    path=args.output/'config.yaml';path.write_text(yaml.safe_dump(config,sort_keys=False))
    from scripts.run_longlive_sparse import run_config
    loaded=run_config(path)['pipeline']
    from pipeline.causal_inference import CausalInferencePipeline
    from pipeline.interactive_causal_inference import InteractiveCausalInferencePipeline
    from utils.misc import set_seed
    original_source=Path(inspect.getfile(InteractiveCausalInferencePipeline))
    records=[];reference=None
    variants=['native_single','interactive_single','cross_only','official_recache'] if args.gate else ['cross_only','official_recache']
    report=dict(status='running',seed=args.seed,latent_frames=length,segments=segments,novel_control=args.novel_control,
        gpu=torch.cuda.get_device_name(),source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        upstream_interactive_source=str(original_source),upstream_interactive_sha256=hashlib.sha256(original_source.read_bytes()).hexdigest(),
        variants=records,all_original_weights_shared=True,history_archive_and_onload=False)
    for name in variants:
        root=args.output/name;root.mkdir();loaded.sparse_history_archive.reset()
        for module in loaded.sparse_history_modules:module.clear_selection_cache();module.clear_capture_state()
        cls=CausalInferencePipeline if name=='native_single' else InteractiveCausalInferencePipeline
        pipeline=cls(loaded.args,torch.device('cuda'),generator=loaded.generator,text_encoder=loaded.text_encoder,vae=loaded.vae)
        recache_events=[]
        if name not in ('native_single','interactive_single'):
            original_recache=pipeline._recache_after_switch
            def recache(owner,output,current_start_frame,new_conditional_dict):
                started=time.perf_counter()
                if name=='cross_only':
                    for cache in owner.crossattn_cache:
                        cache['k'].zero_();cache['v'].zero_();cache['is_init']=False
                else:original_recache(output,current_start_frame,new_conditional_dict)
                torch.cuda.synchronize()
                recache_events.append(dict(at_latent=current_start_frame,wall_s=time.perf_counter()-started,
                    past_frames=min(owner.local_attn_size,current_start_frame) if name=='official_recache' else 0))
            pipeline._recache_after_switch=types.MethodType(recache,pipeline)
        set_seed(args.seed);noise=torch.randn(1,length,16,60,104,device='cuda',dtype=torch.bfloat16)
        begin=time.perf_counter()
        try:
            if name=='native_single':
                video,latent=pipeline.inference(noise=noise,text_prompts=[segments[0]['prompt']],return_latents=True,low_memory=True,profile=False)
            else:
                selected=segments[:1] if name=='interactive_single' else segments
                video,latent=pipeline.inference(noise=noise,text_prompts_list=[[s['prompt']] for s in selected],
                    switch_frame_indices=[s['start_latent'] for s in selected[1:]],return_latents=True,low_memory=True)
            torch.cuda.synchronize();generation_decode_s=time.perf_counter()-begin
            latent=latent.cpu();torch.save(latent,root/'latents.pt')
            sink=IncrementalVideoSink(root/'video.mp4',expected_frames=4*length-3,started=begin,input_range='unit')
            sink(video);pixels=sink.close()
            if not torch.isfinite(latent).all():raise RuntimeError('nonfinite native interactive output')
            identity=dict(noise=tensor_sha256(noise),latent=tensor_sha256(latent),raw_RGB=pixels['raw_RGB_sha256'])
            if name=='native_single':reference=identity
            if name=='interactive_single' and identity!=reference:raise RuntimeError('no-switch native/interactive pipelines differ')
            stats=loaded.sparse_history_archive.stats.as_dict()
            if stats['transferred_bytes'] or stats['archive_bytes']:raise RuntimeError('local interactive baseline acquired history archive')
            record=dict(variant=name,status='pass',identity=identity,video=str(root/'video.mp4'),
                generation_decode_s=generation_decode_s,generation_decode_encode_s=time.perf_counter()-begin,
                recache_events=recache_events,source_switch_recache=name=='official_recache',
                quality='pending_review',history_H2D_bytes=0,archive_bytes=0)
            (root/'stats.json').write_text(json.dumps(stats,indent=2)+'\n');(root/'terminal.json').write_text(json.dumps(record,indent=2)+'\n')
            records.append(record);print(json.dumps(record),flush=True)
            del video,latent,noise,pipeline
        except BaseException:
            failure=dict(variant=name,status='fail',traceback=traceback.format_exc());records.append(failure)
            report['status']='fail';(root/'terminal.json').write_text(json.dumps(failure,indent=2)+'\n')
            (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');raise
        finally:loaded.vae.model.clear_cache();torch.cuda.empty_cache();gc.collect()
        (args.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    report['status']='pass';(args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
