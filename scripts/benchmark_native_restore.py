#!/usr/bin/env python3
"""Fair warm KV restoration: pageable, bounded pinned staging, or clean-log replay.

No new video generation, T5 or VAE loading. All restored positive KV is checked
against full native witness hashes outside timing. This is not end-to-end video.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
CREATED_OUTPUT=None


@torch.inference_mode()
def main():
    global CREATED_OUTPUT
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=30);args=p.parse_args()
    args.case=args.case.resolve();args.assets=args.assets.resolve();args.source=args.source.resolve();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    CREATED_OUTPUT=args.output
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    expected=json.loads((args.case/'inflight_replay_audit.json').read_text())
    original=json.loads((args.case/'summary.json').read_text())
    log=torch.load(args.case/'clean_commit_prefix_log.pt',map_location='cpu',weights_only=True)
    if not expected['full_final_cache_bitwise_exact'] or not original['observer_noise_latent_RGB_equivalence']:
        raise ValueError('verified full native witness required')
    if subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()!=original['upstream_source_SHA']:
        raise ValueError('source changed')
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(args.source));os.chdir(args.assets)
    from omegaconf import OmegaConf
    from pipeline import CausalDiffusionInferencePipeline
    from utils.config import normalize_config
    from utils.wan_5b_wrapper import CausalWanModel
    from utils.inference_utils import load_generator_checkpoint
    from adapters.longlive_sparse.history_cache import tensor_sha256
    from adapters.longlive_sparse.native_commit_replay import owned_cpu
    raw=OmegaConf.load(args.case/'config.yaml');config=normalize_config(raw)
    def architecture(path,**kwargs):return CausalWanModel.from_config(json.loads((Path(path)/'config.json').read_text()),**kwargs)
    started=time.perf_counter()
    with patch.object(CausalWanModel,'from_pretrained',side_effect=architecture):
        pipe=CausalDiffusionInferencePipeline(config,device=torch.device('cuda'),text_encoder=torch.nn.Identity(),vae=torch.nn.Identity())
    matched=load_generator_checkpoint(pipe.generator,str(args.assets/'checkpoints/model_bf16.pt'),strict=True)
    if matched.missing_keys or matched.unexpected_keys:raise RuntimeError('incomplete model')
    pipe.generator.to(device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
    pipe._set_all_modules_max_attention_size(pipe.local_attn_size);pipe._set_all_modules_sink_size(pipe.sink_size)
    pipe._set_all_modules_global_sink_size(pipe.global_sink_size)
    pipe._initialize_kv_cache(1,torch.bfloat16,torch.device('cuda'))
    pipe._initialize_crossattn_cache(1,torch.bfloat16,torch.device('cuda'))
    model_load_s=time.perf_counter()-started
    conditions={k:v.pin_memory() for k,v in log['conditions'].items()}
    records=[dict(r,latent=r['latent'].pin_memory(),timestep=r['timestep'].pin_memory()) for r in log['records']]
    covered=records[-1]['current_start']//pipe.frame_seq_length+records[-1]['latent'].shape[1]
    if covered<pipe.local_attn_size:raise ValueError('metadata-only reset requires full cache coverage in this benchmark')
    witnesses=torch.load(args.case/'offline_sample_witness.pt',map_location='cpu',weights_only=True)
    expected_metadata=witnesses[len(records)-1]['metadata']
    def reset_metadata():
        for c in pipe.kv_cache_pos:
            c['global_end_index'].zero_();c['local_end_index'].zero_();c['pinned_start'].fill_(-1);c['pinned_len'].zero_()
    def replay():
        reset_metadata()
        for i,r in enumerate(records):
            for k,v in r['settings'].items():setattr(pipe._dit_model,k,v)
            for c in pipe.crossattn_cache_pos:c['is_init']=False
            x=r['latent'].cuda();t=r['timestep'].cuda();c=conditions[r['condition_key']].cuda()
            pipe.generator(noisy_image_or_video=x,conditional_dict={'prompt_embeds':c},timestep=t,
                kv_cache=pipe.kv_cache_pos,crossattn_cache=pipe.crossattn_cache_pos,
                current_start=r['current_start'],cache_start=r['cache_start'])
            if pipe._is_scene_cut(original['prompts_per_block'][:len(records)],i):
                pipe._pin_current_chunk(pipe.kv_cache_pos,x.shape[1])
    replay();torch.cuda.synchronize()
    def verify():
        for row,c in zip(expected['full_final_cache_comparisons'],pipe.kv_cache_pos):
            if tensor_sha256(c['k'])!=row['original_K_sha256'] or tensor_sha256(c['v'])!=row['original_V_sha256']:
                raise RuntimeError('restored KV differs from original full witness')
            if any(int(c[k])!=v for k,v in expected_metadata.items()):raise RuntimeError('restored cache metadata differs')
    verify()
    bank=[{k:owned_cpu(c[k]) for k in ('k','v','global_end_index','local_end_index','pinned_start','pinned_len')} for c in pipe.kv_cache_pos]
    staging=torch.empty_like(bank[0]['k'],device='cpu',pin_memory=True)
    pinned=staging.numel()*staging.element_size()+sum(v.numel()*v.element_size() for v in conditions.values())
    pinned+=sum(r[k].numel()*r[k].element_size() for r in records for k in ('latent','timestep'))
    if pinned>128*1024**2:raise RuntimeError('bounded128MiB pinned budget exceeded')
    def raw_restore(use_staging):
        for source,target in zip(bank,pipe.kv_cache_pos):
            for key in ('k','v'):
                if use_staging:
                    staging.copy_(source[key]);target[key].copy_(staging)
                else:target[key].copy_(source[key])
            for key in ('global_end_index','local_end_index','pinned_start','pinned_len'):target[key].copy_(source[key])
    actions={'raw_pageable':lambda:raw_restore(False),'raw_bounded_pinned':lambda:raw_restore(True),'clean_log_replay':replay}
    for action in actions.values():
        for _ in range(5):action()
        torch.cuda.synchronize();verify()
    torch.cuda.reset_peak_memory_stats();samples={k:[] for k in actions};rng=random.Random(20260908);order=[]
    for repeat in range(args.repeats):
        names=list(actions);rng.shuffle(names);order.append(names)
        for name in names:
            torch.cuda.synchronize();begin=time.perf_counter();actions[name]();torch.cuda.synchronize()
            samples[name].append(time.perf_counter()-begin)
    for action in actions.values():action();torch.cuda.synchronize();verify()
    raw_bytes=sum(c[k].numel()*c[k].element_size() for c in bank for k in ('k','v'))
    result=dict(status='pass',scope='warm_positive_KV_restore_only_not_video_or_cold_storage_IO',gpu=torch.cuda.get_device_name(),
        source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        source_case=str(args.case),source_case_summary_sha256=hashlib.sha256((args.case/'summary.json').read_bytes()).hexdigest(),
        clean_log_file_sha256=hashlib.sha256((args.case/'clean_commit_prefix_log.pt').read_bytes()).hexdigest(),
        full_original_KV_hash_gates=True,raw_KV_bytes=raw_bytes,log_tensor_bytes=expected['log_tensor_bytes'],
        common_model_weights_excluded_from_state_bytes=True,host_pinned_bytes=pinned,host_pinned_budget_bytes=128*1024**2,
        cache_preallocated=True,log_reset_metadata_only_safe_for_full_cache_coverage=True,model_load_s=model_load_s,
        first_restore_and_warmup_and_hash_validation_excluded=True,warmup_per_mode=5,repeats=args.repeats,blocked_randomized_order=order,
        medians_s={k:statistics.median(v) for k,v in samples.items()},p95_s={k:float(np.percentile(v,95)) for k,v in samples.items()},
        samples_s=samples,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
        negative_KV_retained_as_native_baseline=True,raw_includes_pack_when_staged=True,
        both_representations_retained_for_controlled_benchmark=True,process_RSS_reduction_not_measured=True,
        raw_does_not_need_historical_text_projections_but_replay_does=True,no_hardware_counter_or_cold_tier_bandwidth_claim=True)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples_s','blocked_randomized_order')}))


if __name__=='__main__':
    try:main()
    except BaseException:
        if CREATED_OUTPUT is not None and not (CREATED_OUTPUT/'summary.json').exists():
            (CREATED_OUTPUT/'summary.json').write_text(json.dumps(dict(status='fail',traceback=traceback.format_exc(),
                partial_artifacts_preserved=True),indent=2)+'\n')
        raise
