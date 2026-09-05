#!/usr/bin/env python3
"""Fresh diagnostic pipeline: late full Attention capture + a separate timeline.

Capture is at latent 30; Nsight covers the first non-classification generator
call at latent 36. No capture occurs inside the profiled call. This is not a
video-quality or end-to-end speed trial. Noise and RNG state are retained.
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
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.system_config import LongLiveSystemConfig


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prompt',choices=('calibration_motion','calibration_state'),required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    if (root/'capture_run.json').exists():
        raise RuntimeError('terminal diagnostic already exists; do not duplicate')
    os.environ['INFER_OUTPUT_DIR']=str(root)
    os.environ['LONGLIVE_CAPTURE_COMPLETE_ATTENTION']='1'
    os.environ['LONGLIVE_COMPLETE_CAPTURE_STARTS']='46800'
    os.environ['LONGLIVE_COMPLETE_CAPTURE_LAYERS']='0,9,19,29'
    os.environ['LONGLIVE_COMPLETE_CAPTURE_PASSES']='1'
    os.environ['LONGLIVE_CAPTURE_CASE_TAG']=args.prompt
    os.environ['LONGLIVE_NVTX']='1'
    prompt=next(item for item in json.loads((ROOT/'configs/system/profile_calibration_prompts.json').read_text())['candidates']
                if item['prompt_id']==args.prompt)
    params=json.loads((ROOT/'configs/formal/method_params.json').read_text())['method_params']['transfer_vaware_hybrid_history']
    config=yaml.safe_load((ROOT/'configs/inferhub/rag_method_21.yaml').read_text())
    (root/'empty_prompts.txt').write_text('')
    config.update(data_path=str(root/'empty_prompts.txt'),output_folder=str(root/'base_load'),inference_iter=0)
    config['sparse_history'].update(method='transfer_vaware_hybrid_history',history_density=.25,method_params=params,record_per_call=True)
    system=LongLiveSystemConfig(profile_mode='trace',transfer_layout='exact_compact',staging_mode='persistent_separate',
        cpu_pack_policy='archive_runs',gpu_union_cache='per_chunk',gpu_union_cache_budget_mib=768)
    config['longlive_system']=system.as_dict()
    config_path=root/'load_config.yaml'
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    from scripts.run_longlive_sparse import run_config
    start=time.perf_counter()
    pipeline=run_config(config_path)['pipeline']
    load_s=time.perf_counter()-start
    gate={'active':False,'completed':False}
    def before(module, positional, keywords):
        if keywords.get('current_start')==56160 and not keywords.get('classify_mode',False) and not gate['completed']:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push('full_generator_late_chunk_first_pass')
            gate['active']=True
    def after(module, positional, keywords, output):
        if gate['active']:
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
            gate.update(active=False,completed=True)
    hooks=[pipeline.generator.register_forward_pre_hook(before,with_kwargs=True),
           pipeline.generator.register_forward_hook(after,with_kwargs=True)]
    from utils.misc import set_seed
    set_seed(20260904)
    device=next(pipeline.generator.parameters()).device
    noise=torch.randn(1,39,16,60,104,device=device,dtype=torch.bfloat16)
    initial_noise_sha=tensor_sha(noise)
    rng=torch.cuda.get_rng_state(device)
    torch.save({'noise':noise.cpu(),'cuda_rng_state_before_inference':rng},root/'input_noise_and_rng.pt')
    start=time.perf_counter()
    with torch.inference_mode():
        _,latent=pipeline.inference(noise=noise,text_prompts=[prompt['prompt']],return_latents=True,
                                    low_memory=True,profile=True,skip_vae_decode=True)
    torch.cuda.synchronize()
    diagnostic_s=time.perf_counter()-start
    for hook in hooks:
        hook.remove()
    if not gate['completed'] or not torch.isfinite(latent).all():
        raise RuntimeError('late trace hook absent or nonfinite latent')
    torch.save(latent.cpu(),root/'latents.pt')
    captures=sorted((root/'complete_attention_captures'/args.prompt).glob('*.pt'))
    if len(captures)!=4:
        raise RuntimeError(f'expected four completed late captures, found {len(captures)}')
    summary=[]
    for path in captures:
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['key'].shape[1]!=9360:
            raise RuntimeError('late diagnostic did not reach six historical candidate frames')
        summary.append({'path':str(path),'layer':payload['layer'],'history_tokens':payload['key'].shape[1],
                        'exact_tokens':payload['exact_key'].shape[1]})
    result={'status':'pass','scope':'late_capture_and_separate_profile_diagnostic_only','prompt':prompt,
        'seed':20260904,'latent_frames':39,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'system':system.as_dict(),'model_load_s':load_s,'diagnostic_pipeline_s':diagnostic_s,
        'capture_start':46800,'trace_start':56160,'capture_inside_profiled_call':False,
        'captures':summary,'initial_noise_sha256':initial_noise_sha,'latent_sha256':tensor_sha(latent),
        'paired_to_previous_video':False,'end_to_end_speed_claim':False,'vae_decoded':False}
    (root/'capture_run.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'sparse_history_stats.json').write_text(json.dumps(pipeline.sparse_history_aggregate_stats.as_dict(),indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
