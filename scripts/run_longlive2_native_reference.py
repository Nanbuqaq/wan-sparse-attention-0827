#!/usr/bin/env python3
"""Small source-locked LongLive2 BF16 reference; never a cross-backbone speedup.

Official model, scheduler and inference loop are unchanged. To avoid downloading
20GB of base weights that would be entirely replaced, initialize its official
architecture from config and STRICTLY load the complete released generator.
Text embeddings are produced by its own T5 before offload; VAE decoding is the
native batch decode after generation. These placement choices are recorded.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
SOURCE_SHA='6b36d20ec6f7958d29d11a704dfa64611a9f2572'


def native_schedule(root, length, control=None):
    spec=json.loads((root/'configs/system/memory_revisit_development.json').read_text())['scenarios'][1]
    starts=(0,8,16) if length==24 else (0,24,48,80)
    segments=[dict(s,start_latent=t) for s,t in zip(spec['segments'],starts)]
    if control:
        controls=json.loads((root/'configs/system/event_retrieval_negative_controls.json').read_text())
        segments[-1]['prompt']=controls['controls'][control]
    prompts=[next(s['prompt'] for s in reversed(segments) if s['start_latent']<=i) for i in range(0,length,8)]
    return segments,[prompts]


class CachedNativeTextEncoder(torch.nn.Module):
    def __init__(self,values,device):
        super().__init__();self.values=values;self.target_device=device

    def forward(self,*,text_prompts):
        if any(p not in self.values for p in text_prompts):raise ValueError('unencoded native prompt')
        return {'prompt_embeds':torch.cat([self.values[p] for p in text_prompts],dim=0).to(self.target_device)}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'third_party/LongLive2')
    p.add_argument('--gate',action='store_true');p.add_argument('--seed',type=int,default=20260909)
    p.add_argument('--control',choices=('duck','empty'));args=p.parse_args()
    args.output=args.output.resolve();args.assets=args.assets.resolve();args.source=args.source.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    source_sha=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    if source_sha!=SOURCE_SHA:raise ValueError('LongLive2 source must be locked')
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    manifest=args.assets/'assets_manifest.json';assets=json.loads(manifest.read_text())
    if assets['status']!='pass':raise ValueError('verified assets required')
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(args.source));os.chdir(args.assets)
    from omegaconf import OmegaConf
    from pipeline import CausalDiffusionInferencePipeline
    from utils.config import normalize_config
    from utils.wan_5b_wrapper import CausalWanModel
    from utils.inference_utils import load_generator_checkpoint
    from adapters.longlive_sparse.history_cache import tensor_sha256
    from adapters.longlive_sparse.stream_video_sink import IncrementalVideoSink
    raw=OmegaConf.load(args.source/'configs/inference.yaml')
    del raw.adapter
    raw.checkpoints.lora_ckpt=None;raw.checkpoints.generator_ckpt=str(args.assets/'checkpoints/model_bf16.pt')
    raw.inference.streaming_vae=False;raw.inference.async_vae=False;raw.inference.vae_device=None
    length=24 if args.gate else 128
    raw.data.image_or_video_shape[1]=length
    if args.gate:raw.model_kwargs.local_attn_size=16
    config=normalize_config(raw)
    segments,prompts=native_schedule(ROOT,length,args.control)
    report=dict(status='running',upstream_source_SHA=source_sha,
        runner_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        assets_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,seed=args.seed,gate=args.gate,latent_frames=length,pixel_frames=4*length-3,
        latent_shape=[1,length,48,44,80],local_frames=16 if args.gate else 32,sink_frames=8,
        native_default_resolution=True,segments=segments,control=args.control,
        source_files_sha256={name:hashlib.sha256((args.source/name).read_bytes()).hexdigest() for name in (
            'pipeline/causal_diffusion_inference.py','utils/wan_5b_wrapper.py','wan_5b/modules/causal_model.py')},
        loading='official_from_config_then_strict_complete_merged_BF16_no_LoRA',
        placement='native_T5_unique_prompts_then_CPU_offload_DiT_GPU_then_CPU_offload_native_VAE_GPU',
        causal_model_and_inference_loop_modified=False,cross_backbone_speedup_claim=False)
    OmegaConf.save(raw,args.output/'config.yaml')
    started=time.perf_counter()
    try:
        def architecture(path,**kwargs):
            cfg=json.loads((Path(path)/'config.json').read_text())
            return CausalWanModel.from_config(cfg,**kwargs)
        with patch.object(CausalWanModel,'from_pretrained',side_effect=architecture):
            pipe=CausalDiffusionInferencePipeline(config,device=torch.device('cuda'))
        # Constructor places native T5 on GPU in FP32. Cast before any inference.
        pipe.text_encoder.to(dtype=torch.bfloat16)
        loaded=load_generator_checkpoint(pipe.generator,str(args.assets/'checkpoints/model_bf16.pt'),strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:raise RuntimeError('incomplete released generator')
        report['strict_generator_load']=dict(missing_keys=loaded.missing_keys,unexpected_keys=loaded.unexpected_keys,
            state_dict_entries=len(pipe.generator.state_dict()))
        report['load_s']=time.perf_counter()-started
        text_started=time.perf_counter();encoded={}
        for prompt in dict.fromkeys(prompts[0]):
            encoded[prompt]=pipe.text_encoder(text_prompts=[prompt])['prompt_embeds'].detach().cpu()
        torch.cuda.synchronize();report['native_T5_s']=time.perf_counter()-text_started
        pipe.text_encoder.to('cpu');pipe.text_encoder=CachedNativeTextEncoder(encoded,torch.device('cuda'))
        gc.collect();torch.cuda.empty_cache()
        pipe.generator.to(device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
        torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        noise=torch.randn(1,length,48,44,80,device='cuda',dtype=torch.bfloat16)
        report['noise_sha256']=tensor_sha256(noise)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();generation_started=time.perf_counter()
        latent=pipe.inference(noise=noise,text_prompts=prompts,return_latents=True)
        torch.cuda.synchronize();report['native_DiT_s']=time.perf_counter()-generation_started
        report['generation_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        if not bool(torch.isfinite(latent).all()):raise RuntimeError('nonfinite native LongLive2 latents')
        report['latent_sha256']=tensor_sha256(latent);torch.save(latent.cpu(),args.output/'latents.pt')
        kv_bytes=sum(v.numel()*v.element_size() for caches in (pipe.kv_cache_pos,pipe.kv_cache_neg) for c in caches for k,v in c.items() if k in ('k','v'))
        report['native_positive_and_negative_KV_bytes']=kv_bytes
        offload_started=time.perf_counter()
        pipe.kv_cache_pos=pipe.kv_cache_neg=pipe.crossattn_cache_pos=pipe.crossattn_cache_neg=None
        pipe.generator.to('cpu');gc.collect();torch.cuda.empty_cache()
        pipe.vae.to(device='cuda',dtype=torch.bfloat16)
        torch.cuda.synchronize();report['generation_to_decode_placement_s']=time.perf_counter()-offload_started
        decode_started=time.perf_counter();video=pipe.vae.decode_to_pixel(latent)
        torch.cuda.synchronize();report['native_VAE_s']=time.perf_counter()-decode_started
        sink=IncrementalVideoSink(args.output/'video.mp4',expected_frames=4*length-3,started=generation_started,fps=24)
        sink(video);report['pixels']=sink.close()
        report.update(status='pass',video=str(args.output/'video.mp4'),wall_including_loading_s=time.perf_counter()-started,
            quality='pending_own_identity_absence_and_transition_review')
    except BaseException:
        report.update(status='fail',traceback=traceback.format_exc(),partial_artifacts_preserved=True)
        raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('segments','source_files_sha256','traceback')}),flush=True)


if __name__=='__main__':main()
