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
CREATED_OUTPUT=None


def native_schedule(root, length, control=None):
    spec=json.loads((root/'configs/system/memory_revisit_development.json').read_text())['scenarios'][1]
    starts=(0,8,16) if length==24 else (0,24,48,80)
    segments=[dict(s,start_latent=t) for s,t in zip(spec['segments'],starts)]
    if control:
        controls=json.loads((root/'configs/system/event_retrieval_negative_controls.json').read_text())
        segments[-1]['prompt']=controls['controls'][control]
    prompts=[next(s['prompt'] for s in reversed(segments) if s['start_latent']<=i) for i in range(0,length,8)]
    return segments,[prompts]


def native_cut_schedule(root,scenario,*,gate=False,episode_gate=False):
    spec=json.loads((root/'configs/system/native_cut_memory_development.json').read_text())
    selected=next(s for s in spec['scenarios'] if s['id']==scenario)
    starts=((0,8,16,48) if episode_gate else (0,8,16,32)) if gate else (0,24,48,96)
    segments=[dict(s,start_latent=t) for s,t in zip(selected['segments'],starts)]
    length=(64 if episode_gate else 48) if gate else spec['latent_frames'];prompts=[]
    for frame in range(0,length,8):
        i=max(i for i,s in enumerate(segments) if s['start_latent']<=frame)
        prefix=spec['native_scene_cut_prefix'] if i>0 and frame==segments[i]['start_latent'] else ''
        prompts.append(prefix+segments[i]['prompt'])
    return segments,[prompts]


class CachedNativeTextEncoder(torch.nn.Module):
    def __init__(self,values,device):
        super().__init__();self.values=values;self.target_device=device

    def forward(self,*,text_prompts):
        if any(p not in self.values for p in text_prompts):raise ValueError('unencoded native prompt')
        return {'prompt_embeds':torch.cat([self.values[p] for p in text_prompts],dim=0).to(self.target_device)}


@torch.inference_mode()
def main():
    global CREATED_OUTPUT
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'third_party/LongLive2')
    p.add_argument('--gate',action='store_true');p.add_argument('--seed',type=int,default=20260909)
    p.add_argument('--cut-scenario',choices=('generated_patchwork_toy_cut_revisit','generated_bead_state_cut_revisit'))
    p.add_argument('--audit-clean-replay',action='store_true')
    p.add_argument('--equivalence-reference',type=Path)
    p.add_argument('--replay-resume-after-latents',type=int,default=0)
    p.add_argument('--episode-memory-mode',choices=('none','raw_reveal','raw_away','log_reveal'))
    p.add_argument('--control',choices=('duck','empty'));args=p.parse_args()
    args.output=args.output.resolve();args.assets=args.assets.resolve();args.source=args.source.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    CREATED_OUTPUT=args.output
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    source_sha=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    if source_sha!=SOURCE_SHA:raise ValueError('LongLive2 source must be locked')
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    for key in ('LLV2_USE_FA3','LLV2_USE_FA4','LLV2_USE_TE_ATTN'):
        if os.environ.get(key,'0')!='0':raise ValueError('this reference is fixed to native BF16 FA2')
    manifest=args.assets/'assets_manifest.json';assets=json.loads(manifest.read_text())
    if assets['status']!='pass':raise ValueError('verified assets required')
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(args.source));os.chdir(args.assets)
    from omegaconf import OmegaConf
    from pipeline import CausalDiffusionInferencePipeline
    from utils.config import normalize_config
    from utils.wan_5b_wrapper import CausalWanModel
    from utils.inference_utils import load_generator_checkpoint
    import wan_5b.modules.attention as native_attention
    if not native_attention.FLASH_ATTN_2_AVAILABLE:raise RuntimeError('native FA2 required, no SDPA fallback')
    from adapters.longlive_sparse.history_cache import tensor_sha256
    from adapters.longlive_sparse.stream_video_sink import IncrementalVideoSink
    raw=OmegaConf.load(args.source/'configs/inference.yaml')
    del raw.adapter
    raw.checkpoints.lora_ckpt=None;raw.checkpoints.generator_ckpt=str(args.assets/'checkpoints/model_bf16.pt')
    raw.inference.streaming_vae=False;raw.inference.async_vae=False;raw.inference.vae_device=None
    length=24 if args.gate else 128
    if args.cut_scenario and args.control:raise ValueError('new cut feasibility is not an old negative control')
    if args.cut_scenario and args.gate:
        length=48;raw.data.image_or_video_shape[-2:]=[32,56]
    if args.episode_memory_mode is not None:
        if not args.cut_scenario or args.audit_clean_replay or args.equivalence_reference:
            raise ValueError('episode intervention is a separate cut-workload experiment')
        if args.gate:
            length=64;raw.data.image_or_video_shape[-2:]=[16,32]
    raw.data.image_or_video_shape[1]=length
    if args.gate and not args.cut_scenario:raw.model_kwargs.local_attn_size=16
    config=normalize_config(raw)
    segments,prompts=(native_cut_schedule(ROOT,args.cut_scenario,gate=args.gate,episode_gate=args.episode_memory_mode is not None) if args.cut_scenario
                     else native_schedule(ROOT,length,args.control))
    latent_height,latent_width=map(int,raw.data.image_or_video_shape[-2:])
    report=dict(status='running',upstream_source_SHA=source_sha,
        runner_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        assets_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,seed=args.seed,gate=args.gate,latent_frames=length,pixel_frames=4*length-3,
        latent_shape=[1,length,48,latent_height,latent_width],local_frames=int(raw.model_kwargs.local_attn_size),sink_frames=8,
        native_default_resolution=(latent_height,latent_width)==(44,80),segments=segments,control=args.control,
        cut_scenario=args.cut_scenario,prompts_per_block=prompts[0],
        expected_scene_cut_block_indices=[i for i,x in enumerate(prompts[0]) if x.startswith('The scene transitions. ')],
        source_files_sha256={name:hashlib.sha256((args.source/name).read_bytes()).hexdigest() for name in (
            'pipeline/causal_diffusion_inference.py','utils/wan_5b_wrapper.py','wan_5b/modules/causal_model.py')},
        loading='official_from_config_then_strict_complete_merged_BF16_no_LoRA',
        placement='native_T5_unique_prompts_then_CPU_offload_DiT_GPU_then_CPU_offload_native_VAE_GPU',
        causal_model_and_inference_loop_modified=False,cross_backbone_speedup_claim=False,
        attention_backend='native_FA2',KV_and_generator_dtype='bfloat16',fallback_allowed=False,
        non_FA2_backends_disabled=True)
    report['capture_augmented_clean_replay']=args.audit_clean_replay
    report['episode_memory_mode']=args.episode_memory_mode
    if args.replay_resume_after_latents:
        if not args.audit_clean_replay or not 0<args.replay_resume_after_latents<length or args.replay_resume_after_latents%8:
            raise ValueError('inflight replay needs a strict interior block-aligned boundary and audit flag')
        if args.replay_resume_after_latents in [(i+1)*8 for i in report['expected_scene_cut_block_indices']]:
            raise ValueError('this first inflight gate pauses after a non-cut commit, not before a pending pin')
    external=None
    if args.equivalence_reference:
        external=json.loads(args.equivalence_reference.read_text())
        if any(external[k]!=report[k] for k in ('seed','latent_shape','prompts_per_block')) or external['status']!='pass':
            raise ValueError('observer reference identity differs')
    OmegaConf.save(raw,args.output/'config.yaml')
    started=time.perf_counter()
    (args.output/'progress.json').write_text(json.dumps(dict(report,stage='native_model_initialization'),indent=2)+'\n')
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
        (args.output/'progress.json').write_text(json.dumps(dict(report,stage='native_text_encoding'),indent=2)+'\n')
        text_started=time.perf_counter();encoded={}
        for prompt in dict.fromkeys(prompts[0]):
            encoded[prompt]=pipe.text_encoder(text_prompts=[prompt])['prompt_embeds'].detach().cpu()
        torch.cuda.synchronize();report['native_T5_s']=time.perf_counter()-text_started
        pipe.text_encoder.to('cpu');pipe.text_encoder=CachedNativeTextEncoder(encoded,torch.device('cuda'))
        gc.collect();torch.cuda.empty_cache()
        pipe.generator.to(device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
        torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        noise=torch.randn(1,length,48,latent_height,latent_width,device='cuda',dtype=torch.bfloat16)
        report['noise_sha256']=tensor_sha256(noise)
        pin_events=[]
        if args.cut_scenario:
            original_pin=pipe._pin_current_chunk
            def observe_pin(caches,current_num_frames):
                original_pin(caches,current_num_frames)
                pin_events.append(dict(completed_latent=int(caches[0]['global_end_index'])//pipe.frame_seq_length,
                    pinned_start=int(caches[0]['pinned_start']),pinned_tokens=int(caches[0]['pinned_len'])))
            pipe._pin_current_chunk=observe_pin
        replay_log=replay_hook=None;inflight_audit=None;episode_memory=None
        if args.episode_memory_mode is not None:
            from adapters.longlive_sparse.native_episode_memory import NativeEpisodeMemory
            episode_memory=NativeEpisodeMemory(pipe,mode=args.episode_memory_mode,
                source_end=segments[2]['start_latent'],target_start=segments[-1]['start_latent'],prompts=prompts[0])
            episode_memory.attach()
        if args.audit_clean_replay:
            from adapters.longlive_sparse.native_commit_replay import NativeCleanCommitLog
            replay_log=NativeCleanCommitLog(pipe)
            def capture_clean(owner,values,kwargs,result):
                nonlocal replay_hook,inflight_audit
                before=len(replay_log.records);replay_log.hook(owner,values,kwargs,result)
                if len(replay_log.records)==before or not args.replay_resume_after_latents or inflight_audit is not None:return
                end=int(kwargs['current_start'])//pipe.frame_seq_length+kwargs['noisy_image_or_video'].shape[1]
                if end!=args.replay_resume_after_latents:return
                replay_hook.remove()
                pin_function=pipe._pin_current_chunk
                if args.cut_scenario:pipe._pin_current_chunk=original_pin
                torch.save(replay_log.payload(),args.output/'clean_commit_prefix_log.pt')
                prefix=torch.cat([r['latent'] for r in replay_log.records],dim=1).to(noise.device)
                with torch.random.fork_rng(devices=[noise.device]):
                    inflight_audit=replay_log.replay_and_compare(prompts=prompts[0][:len(replay_log.records)],returned_latent=prefix)
                inflight_audit.update(paused_after_latents=end,only_committed_prefix_available=True,
                    future_latents_not_read=True,RNG_state_isolated=True,
                    prefix_log_file_bytes=(args.output/'clean_commit_prefix_log.pt').stat().st_size)
                (args.output/'inflight_replay_audit.json').write_text(json.dumps(inflight_audit,indent=2)+'\n')
                if not inflight_audit['full_final_cache_bitwise_exact']:
                    raise RuntimeError('inflight rematerialization is not exact; no hidden original-KV fallback')
                pipe._pin_current_chunk=pin_function
                replay_hook=pipe.generator.register_forward_hook(capture_clean,with_kwargs=True)
            replay_hook=pipe.generator.register_forward_hook(capture_clean,with_kwargs=True)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();generation_started=time.perf_counter()
        (args.output/'progress.json').write_text(json.dumps(dict(report,stage='native_generation'),indent=2)+'\n')
        latent=pipe.inference(noise=noise,text_prompts=prompts,return_latents=True)
        torch.cuda.synchronize();report['native_DiT_s']=time.perf_counter()-generation_started
        if episode_memory is not None:
            episode_memory.detach();report['episode_memory']=episode_memory.audit()
            report['pre_return_latent_sha256']=tensor_sha256(latent[:,:segments[-1]['start_latent']])
        report['native_shot_pin_events']=list(pin_events)
        if args.cut_scenario:
            expected=[(i+1)*8 for i in report['expected_scene_cut_block_indices']]
            if [r['completed_latent'] for r in pin_events]!=expected:raise RuntimeError('native scene-cut pin branches did not execute as declared')
        report['generation_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        if not bool(torch.isfinite(latent).all()):raise RuntimeError('nonfinite native LongLive2 latents')
        report['latent_sha256']=tensor_sha256(latent);torch.save(latent.cpu(),args.output/'latents.pt')
        kv_bytes=sum(v.numel()*v.element_size() for caches in (pipe.kv_cache_pos,pipe.kv_cache_neg) for c in caches for k,v in c.items() if k in ('k','v'))
        report['native_positive_and_negative_KV_bytes']=kv_bytes
        report['final_native_pinned_slots']=[dict(layer=i,start=int(c['pinned_start']),length=int(c['pinned_len'])) for i,c in enumerate(pipe.kv_cache_pos)]
        if replay_log is not None:
            replay_hook.remove()
            if args.cut_scenario:pipe._pin_current_chunk=original_pin
            torch.save(replay_log.payload(),args.output/'clean_commit_log.pt')
            torch.save([dict(samples=r['samples'],metadata=r['metadata']) for r in replay_log.records],
                args.output/'offline_sample_witness.pt')
            report['clean_commit_replay_audit']=(inflight_audit if inflight_audit is not None else
                replay_log.replay_and_compare(prompts=prompts[0],returned_latent=latent))
            if args.replay_resume_after_latents and inflight_audit is None:raise RuntimeError('inflight boundary not exercised')
            report['clean_commit_replay_audit']['serialized_log_bytes']=(args.output/'clean_commit_log.pt').stat().st_size
            (args.output/'clean_commit_replay_audit.json').write_text(json.dumps(report['clean_commit_replay_audit'],indent=2)+'\n')
        offload_started=time.perf_counter()
        pipe.kv_cache_pos=pipe.kv_cache_neg=pipe.crossattn_cache_pos=pipe.crossattn_cache_neg=None
        pipe.generator.to('cpu');gc.collect();torch.cuda.empty_cache()
        pipe.vae.to(device='cuda',dtype=torch.bfloat16)
        torch.cuda.synchronize();report['generation_to_decode_placement_s']=time.perf_counter()-offload_started
        decode_started=time.perf_counter();video=pipe.vae.decode_to_pixel(latent)
        torch.cuda.synchronize();report['native_VAE_s']=time.perf_counter()-decode_started
        sink=IncrementalVideoSink(args.output/'video.mp4',expected_frames=4*length-3,started=generation_started,fps=24)
        sink(video);report['pixels']=sink.close()
        if external is not None:
            for key in ('noise_sha256','latent_sha256'):
                if report[key]!=external[key]:raise RuntimeError('observer changed generated trajectory')
            if report['pixels']['raw_RGB_sha256']!=external['pixels']['raw_RGB_sha256']:raise RuntimeError('observer/replay changed decoded RGB')
            report['observer_noise_latent_RGB_equivalence']=True
        report.update(status='pass',video=str(args.output/'video.mp4'),wall_including_loading_s=time.perf_counter()-started,
            quality='pending_own_identity_absence_and_transition_review')
    except BaseException:
        report.update(status='fail',traceback=traceback.format_exc(),partial_artifacts_preserved=True)
        raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('segments','source_files_sha256','traceback')}),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException:
        if CREATED_OUTPUT is not None and not (CREATED_OUTPUT/'summary.json').exists():
            with (CREATED_OUTPUT/'summary.json').open('x') as handle:
                json.dump(dict(status='fail',stage='preflight_or_import',traceback=traceback.format_exc(),
                    partial_artifacts_preserved=True),handle,indent=2);handle.write('\n')
        raise
