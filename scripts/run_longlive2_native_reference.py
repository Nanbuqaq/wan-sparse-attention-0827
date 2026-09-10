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
sys.path.insert(0,str(ROOT))
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


def native_cut_schedule(root,scenario,*,gate=False,episode_gate=False,object_text_control=None):
    from adapters.longlive_sparse.object_state_protocol import SCENARIOS,expand_object_scenario
    config_name=('native_object_state_screen.json' if scenario in SCENARIOS else
                 'native_blue_canvas_screen.json' if scenario.startswith('blue_canvas_') else
                 'native_settled_state_screen.json' if scenario.startswith('settled_bead_') else 'native_cut_memory_development.json')
    spec=json.loads((root/'configs/system'/config_name).read_text())
    variant=next((v for v in spec.get('continuation_variants',[]) if v['id']==scenario),None)
    wording=next((v for v in spec.get('wording_variants',[]) if v['id']==scenario),None)
    selected=next(s for s in spec['scenarios'] if s['id']==(variant['base'] if variant else wording['base'] if wording else scenario))
    if config_name=='native_object_state_screen.json':selected=expand_object_scenario(spec,scenario)
    if object_text_control:
        from adapters.longlive_sparse.object_state_protocol import apply_past_text_control
        registration=json.loads((root/'configs/system/native_chest_text_control.json').read_text())
        if object_text_control!=registration['id'] or scenario not in registration['scenarios']:
            raise ValueError('text-control schedule is not registered for this scenario')
        selected=apply_past_text_control(selected,registration)
    if wording:
        selected=dict(selected,segments=[dict(s) for s in selected['segments']])
        targets=[s for s in selected['segments'] if s['start_latent']==wording['at_latent']]
        if len(targets)!=1 or not targets[0]['prompt'].startswith(wording['old_prefix']):raise ValueError('wording control no longer matches original clause')
        targets[0]['prompt']=wording['new_prefix']+targets[0]['prompt'][len(wording['old_prefix']):]
    if variant:
        selected=dict(selected,segments=[dict(s) for s in selected['segments']])
        hold=next(s['prompt'] for s in selected['segments'] if s['role']=='settled_source')
        for segment in selected['segments']:
            if segment['start_latent']>=variant['from_latent']:
                segment['scene_cut']=False
                segment['role']='continuous_explicit' if variant['repeat_settled_source_prompt'] else 'continuous_anaphora'
                if variant['repeat_settled_source_prompt']:segment['prompt']=hold
    if gate and config_name in ('native_settled_state_screen.json','native_blue_canvas_screen.json','native_object_state_screen.json'):
        raise ValueError('new-state feasibility uses original full-length workload only')
    starts=((0,8,16,48) if episode_gate else (0,8,16,32)) if gate else tuple(s['start_latent'] for s in selected['segments'])
    segments=[dict(s,start_latent=t) for s,t in zip(selected['segments'],starts)]
    length=(64 if episode_gate else 48) if gate else spec['latent_frames'];prompts=[]
    for frame in range(0,length,8):
        i=max(i for i,s in enumerate(segments) if s['start_latent']<=frame)
        prefix=spec['native_scene_cut_prefix'] if segments[i].get('scene_cut',i>0) and frame==segments[i]['start_latent'] else ''
        prompts.append(prefix+segments[i]['prompt'])
    return segments,[prompts]


def episode_source_and_target(segments):
    away=[s['start_latent'] for s in segments if s['role']=='away']
    target=[s['start_latent'] for s in segments if s['role']=='return_without_restatement']
    if len(away)!=1 or len(target)!=1 or target[0]-away[0]<32:
        raise ValueError('episode memory requires one committed source and a long away interval')
    return away[0],target[0]


def reviewed_settled_memory_protocol(root,args):
    identifier=getattr(args,'reviewed_memory_protocol',None)
    if identifier is None:return None
    path=root/'configs/system/native_settled_memory_probe.json'
    spec=json.loads(path.read_text())
    if identifier!=spec['id'] or args.cut_scenario!=spec['scenario'] or args.seed not in spec['seeds']:
        raise ValueError('case is outside the reviewed settled-source development protocol')
    if (args.gate or args.episode_memory_mode!='raw_reveal' or args.episode_destination!='shot'
        or args.episode_position_policy not in spec['policies'] or args.scene_context_reset
        or args.memory_reconstruction!='none' or args.episode_restore_after_frames
        or args.cut_component_ablation!='none' or args.native_local_frames!=32
        or not args.cfg1_positive_cache_only):
        raise ValueError('reviewed protocol permits only full-size original/recent raw-shot controls')
    return dict(spec=spec,config_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class CachedNativeTextEncoder(torch.nn.Module):
    def __init__(self,values,device,aliases=None):
        super().__init__();self.values=values;self.target_device=device;self.aliases=aliases or {}

    def forward(self,*,text_prompts):
        text_prompts=[self.aliases.get(p,p) for p in text_prompts]
        if any(p not in self.values for p in text_prompts):raise ValueError('unencoded native prompt')
        return {'prompt_embeds':torch.cat([self.values[p] for p in text_prompts],dim=0).to(self.target_device)}


def validate_causal_runtime_protocol(args,object_protocol):
    if not args.causal_scene_memory:return
    legacy=args.cut_scenario in ('generated_bead_state_cut_revisit','generated_patchwork_toy_cut_revisit','settled_bead_revisit')
    registered=(object_protocol is not None and args.object_state_memory_study and not object_protocol['Dense_only'])
    if (not (legacy or registered)
        or args.episode_memory_mode is not None or args.scene_context_reset or args.memory_reconstruction!='none'
        or args.cut_component_ablation!='none' or args.capture_attention_teacher or args.audit_clean_replay
        or args.reviewed_memory_protocol is not None or args.native_local_frames!=32 or not args.cfg1_positive_cache_only):
        raise ValueError('causal scene baseline requires its isolated qualified native32 protocol')


@torch.inference_mode()
def main():
    global CREATED_OUTPUT
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'third_party/LongLive2')
    p.add_argument('--gate',action='store_true');p.add_argument('--seed',type=int,default=20260909)
    p.add_argument('--cut-scenario',choices=('generated_patchwork_toy_cut_revisit','generated_bead_state_cut_revisit','settled_bead_revisit','settled_bead_visible_control','settled_bead_nocut_anaphora','settled_bead_nocut_explicit','blue_canvas_revisit','blue_canvas_visible_control','blue_canvas_positive_stop_revisit','blue_canvas_positive_stop_visible_control','chest_revisit','chest_visible_control','envelope_revisit','envelope_visible_control'))
    p.add_argument('--audit-clean-replay',action='store_true')
    p.add_argument('--equivalence-reference',type=Path)
    p.add_argument('--replay-resume-after-latents',type=int,default=0)
    p.add_argument('--episode-memory-mode',choices=('none','raw_reveal','raw_away','log_reveal'))
    p.add_argument('--episode-destination',choices=('global','shot'),default='global')
    p.add_argument('--episode-restore-after-frames',type=int,choices=(0,8),default=0)
    p.add_argument('--native-local-frames',type=int,choices=(32,64,96,128))
    p.add_argument('--episode-gate-layout',action='store_true',help='64 latent low-resolution technical workload, including unmodified large-window controls')
    p.add_argument('--cfg1-positive-cache-only',action='store_true')
    p.add_argument('--scene-context-reset',action='store_true')
    p.add_argument('--capture-attention-teacher',action='store_true')
    p.add_argument('--initial-anchor-policy',choices=('keep','source_only','source_repeat','source_repeat_pinned'),default='keep')
    p.add_argument('--memory-reconstruction',choices=('none','past','current'),default='none')
    p.add_argument('--cut-component-ablation',choices=('none','strip_words','freeze_rope','strip_words_freeze_rope'),default='none')
    p.add_argument('--episode-position-policy',choices=('original','recent_virtual','phase_only','age_only'),default='original')
    p.add_argument('--reviewed-memory-protocol',choices=('settled_state_v1',))
    p.add_argument('--causal-scene-memory',action='store_true')
    p.add_argument('--causal-scene-position-policy',choices=('original','recent_virtual'),default=None,
                   help='explicit position-control experiment; omitted keeps the hash-locked original controller')
    p.add_argument('--object-state-memory-study',action='store_true')
    p.add_argument('--object-state-text-control',choices=('past_settled_restatement',))
    p.add_argument('--chest-hybrid-study',action='store_true')
    p.add_argument('--chest-source-pin-lease',action='store_true')
    p.add_argument('--chest-layer-role-probe',action='store_true')
    p.add_argument('--native-numeric-witness',action='store_true')
    p.add_argument('--native-inplace-cache',action='store_true')
    p.add_argument('--causal-block-policy',choices=('full','random','mass_value','contrast_value'))
    p.add_argument('--causal-block-fraction',type=float,default=1.)
    p.add_argument('--causal-block-grouping',choices=('flat64','spatial8','flat_matched'),default='flat64')
    p.add_argument('--causal-block-heads',choices=('shared','per_head'),default='shared')
    p.add_argument('--resident-history-policy', choices=('identity','mass_value','contrast_value','recent'))
    p.add_argument('--resident-history-fraction', type=float, default=.25)
    p.add_argument('--resident-history-reuse', choices=('none','denoise_first'), default='none')
    p.add_argument('--constructor-mode',choices=('reference','strict_checkpoint_no_parameter_init'),default='reference',
        help='experimental common loading path; must pass separate output-equivalence gates')
    p.add_argument('--object-protocol-only',action='store_true',
        help='read-only CPU check of parsed object-state protocol and schedule; no output directory or model')
    p.add_argument('--pipeline-mode',choices=('none','serial','overlap'),default='none')
    p.add_argument('--pipeline-encode-mode',choices=('inline','thread'),default='inline')
    p.add_argument('--pipeline-pixel-slots',type=int,default=2)
    p.add_argument('--pipeline-slots',type=int,default=2)
    p.add_argument('--pipeline-pinned-mib',type=int,default=128)
    p.add_argument('--pipeline-profile',action='store_true',help='NVTX and cudaProfilerApi around real pipeline delivery')
    p.add_argument('--generation-profile',action='store_true',help='profile native generation on its own device; decode remains fully charged separately')
    p.add_argument('--fixed-adaln-warps',type=int,choices=(4,8,16))
    p.add_argument('--fixed-adaln-stages',type=int,choices=(1,2,3),default=1)
    p.add_argument('--control',choices=('duck','empty'));args=p.parse_args()
    from adapters.longlive_sparse.object_state_protocol import validate_object_state_screen
    object_state_screen=validate_object_state_screen(args,ROOT)
    if args.object_state_memory_study and (object_state_screen is None or not args.causal_scene_memory):
        raise ValueError('registered object-state memory study requires its approved scenario and causal memory')
    if args.object_state_text_control and object_state_screen is None:
        raise ValueError('text control requires its registered object-state scenario')
    if args.chest_hybrid_study and (object_state_screen is None or object_state_screen['hybrid_registration'] is None):
        raise ValueError('hybrid study requires the registered factorial protocol')
    if args.chest_source_pin_lease and (object_state_screen is None or object_state_screen['source_pin_lease_registration'] is None):
        raise ValueError('source pin lease requires its explicit registered hybrid protocol')
    if args.chest_layer_role_probe and (object_state_screen is None or object_state_screen['layer_role_probe_registration'] is None or not args.equivalence_reference):
        raise ValueError('all-layer offline probe requires its registered hybrid and full-output reference')
    if not args.causal_scene_memory and args.causal_scene_position_policy is not None:
        raise ValueError('causal position policy requires causal scene memory')
    validate_causal_runtime_protocol(args,object_state_screen)
    if args.object_protocol_only:
        if object_state_screen is None:raise ValueError('object protocol check needs a registered object-state case')
        segments,prompts=native_cut_schedule(ROOT,args.cut_scenario,gate=args.gate,episode_gate=args.episode_gate_layout,
                                            object_text_control=args.object_state_text_control)
        print(json.dumps(dict(status='pass',protocol=object_state_screen,blocks=len(prompts[0]),
            latent_frames=8*len(prompts[0]),scene_cuts=[i for i,prompt in enumerate(prompts[0]) if prompt.startswith('The scene transitions. ')],
            model_or_GPU_initialized=False,output_created=False)))
        return
    args.output=args.output.resolve();args.assets=args.assets.resolve();args.source=args.source.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    CREATED_OUTPUT=args.output
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    source_sha=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    if source_sha!=SOURCE_SHA:raise ValueError('LongLive2 source must be locked')
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    if args.pipeline_profile and args.pipeline_mode=='none':raise ValueError('pipeline profile requires a real pipeline mode')
    if args.pipeline_encode_mode!='inline' and args.pipeline_mode=='none':raise ValueError('pixel worker needs the native pipeline')
    if args.pipeline_mode!='none':
        if torch.cuda.device_count()!=2 or not os.environ.get('WAN_SPARSE_PHYSICAL_GPUS'):
            raise RuntimeError('two physically locked visible GPUs required for pipeline')
        if args.audit_clean_replay or args.capture_attention_teacher or args.memory_reconstruction!='none' or args.episode_memory_mode not in (None,'none'):
            raise ValueError('first pipeline protocol excludes replay/teacher/manual memory branches')
        if args.pipeline_slots<1 or args.pipeline_pinned_mib<1:raise ValueError('explicit positive pipeline budgets required')
    for key in ('LLV2_USE_FA3','LLV2_USE_FA4','LLV2_USE_TE_ATTN'):
        if os.environ.get(key,'0')!='0':raise ValueError('this reference is fixed to native BF16 FA2')
    manifest=args.assets/'assets_manifest.json';assets=json.loads(manifest.read_text())
    if assets['status']!='pass':raise ValueError('verified assets required')
    sys.path.insert(0,str(args.source));os.chdir(args.assets)
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
    if args.cut_scenario and args.cut_scenario.startswith('blue_canvas_') and (args.episode_memory_mode is not None or args.causal_scene_memory or args.memory_reconstruction!='none'):
        raise ValueError('blue canvas is Dense-only until independent source feasibility is reviewed')
    if args.cut_component_ablation!='none' and (args.cut_scenario!='settled_bead_visible_control' or args.episode_memory_mode is not None):
        raise ValueError('cut-component probes are isolated Dense visible controls')
    if args.episode_position_policy!='original' and (args.episode_memory_mode not in ('raw_reveal','raw_away') or args.scene_context_reset or args.episode_destination!='shot'):
        raise ValueError('historical K retiming is an isolated raw-shot source intervention')
    reviewed_protocol=reviewed_settled_memory_protocol(ROOT,args)
    if args.cut_scenario and args.cut_scenario.startswith('settled_bead_') and (args.episode_memory_mode is not None or args.memory_reconstruction!='none') and reviewed_protocol is None:
        raise ValueError('settled-state prompts are Dense-only until feasibility is reviewed and frozen')
    if args.cut_scenario and args.gate:
        length=48;raw.data.image_or_video_shape[-2:]=[32,56]
    episode_layout=args.episode_memory_mode is not None or args.episode_gate_layout
    if args.episode_gate_layout and (not args.gate or not args.cut_scenario):
        raise ValueError('episode gate layout requires a cut-scenario technical gate')
    if args.episode_restore_after_frames and args.episode_memory_mode!='raw_reveal':
        raise ValueError('lifetime intervention requires raw reveal')
    if args.initial_anchor_policy!='keep' and not args.scene_context_reset:
        raise ValueError('initial anchor policy requires a declared logical context reset')
    if args.memory_reconstruction!='none' and (args.initial_anchor_policy!='source_repeat_pinned' or not args.scene_context_reset):
        raise ValueError('semantic reconstruction is a separate repeated-source context policy')
    required_destination='shot' if args.initial_anchor_policy=='keep' else 'global'
    if args.scene_context_reset and (args.episode_memory_mode not in ('raw_reveal','raw_away') or args.episode_destination!=required_destination or args.episode_restore_after_frames):
        raise ValueError('scene context reset requires raw reveal/away in shot role without TTL')
    if args.episode_memory_mode is not None:
        if not args.cut_scenario or args.audit_clean_replay or (args.equivalence_reference and not args.capture_attention_teacher):
            raise ValueError('episode intervention is a separate cut-workload experiment')
    if args.gate and episode_layout:
        length=64;raw.data.image_or_video_shape[-2:]=[16,32]
    raw.data.image_or_video_shape[1]=length
    if args.gate and not args.cut_scenario:raw.model_kwargs.local_attn_size=16
    if args.native_local_frames is not None:
        if args.episode_memory_mode is not None and args.native_local_frames!=32:
            raise ValueError('window baseline cannot attach the local32 admission intervention')
        raw.model_kwargs.local_attn_size=args.native_local_frames
        raw.inference.local_attn_size=args.native_local_frames
    config=normalize_config(raw)
    segments,prompts=(native_cut_schedule(ROOT,args.cut_scenario,gate=args.gate,episode_gate=episode_layout,
                                        object_text_control=args.object_state_text_control) if args.cut_scenario
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
        constructor_mode=args.constructor_mode,
        placement='native_T5_unique_prompts_then_CPU_offload_DiT_GPU_then_CPU_offload_native_VAE_GPU',
        causal_model_and_inference_loop_modified=False,cross_backbone_speedup_claim=False,
        attention_backend='native_FA2',KV_and_generator_dtype='bfloat16',fallback_allowed=False,
        non_FA2_backends_disabled=True)
    report['capture_augmented_clean_replay']=args.audit_clean_replay
    if args.pipeline_mode!='none':
        report.update(pipeline_mode=args.pipeline_mode,pipeline_slots=args.pipeline_slots,pipeline_pinned_budget_bytes=args.pipeline_pinned_mib*1024**2,
            physical_GPU_mapping=os.environ['WAN_SPARSE_PHYSICAL_GPUS'],vae_GPU=torch.cuda.get_device_name(1),
            placement='native_T5_then_DiT_GPU0_and_native_VAE_GPU1_bounded_pinned_pipeline')
    report['episode_memory_mode']=args.episode_memory_mode
    report['episode_restore_after_frames']=args.episode_restore_after_frames
    report['native_window_override']=args.native_local_frames
    report['gpu_total_memory_bytes']=torch.cuda.get_device_properties(0).total_memory
    report['native_KV_allocation_policy']='CFG1_positive_only' if args.cfg1_positive_cache_only else 'native_positive_and_negative'
    report['torch_dynamo_disabled']=os.environ.get('TORCHDYNAMO_DISABLE','0')=='1'
    import triton
    report['triton_version']=triton.__version__
    report['scene_context_reset']=args.scene_context_reset
    report['capture_augmented_attention_teacher']=args.capture_attention_teacher
    report['initial_anchor_policy']=args.initial_anchor_policy
    report['memory_reconstruction']=args.memory_reconstruction
    report['cut_component_ablation']=args.cut_component_ablation
    report['episode_position_policy']='causal_scene_recent_virtual' if args.causal_scene_memory else args.episode_position_policy
    if args.causal_scene_memory:report['causal_scene_memory_enabled']=True
    if reviewed_protocol is not None:report['reviewed_memory_protocol']=reviewed_protocol
    report['fixed_native_adaln_recipe']=(dict(num_warps=args.fixed_adaln_warps,num_stages=args.fixed_adaln_stages)
        if args.fixed_adaln_warps is not None else None)
    if args.fixed_adaln_warps is not None:
        from adapters.longlive_sparse.native_kernel_recipe import fix_native_adaln_for_fresh_run
        fix_native_adaln_for_fresh_run(args.fixed_adaln_warps,args.fixed_adaln_stages)
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
    if object_state_screen is not None:
        report['object_state_protocol']=object_state_screen
        if object_state_screen['Dense_only'] and not args.object_state_text_control:
            report['object_state_dense_screen']=object_state_screen
        if args.chest_hybrid_study:
            report['chest_hybrid_study']=object_state_screen['hybrid_registration']
            report['object_state_text_factor_registration']=object_state_screen['text_control_registration']
        elif args.object_state_text_control:report['object_state_text_control']=object_state_screen['text_control_registration']
    report['causal_scene_position_policy']=(args.causal_scene_position_policy or 'recent_virtual') if args.causal_scene_memory else None
    report['explicit_causal_position_control']=args.causal_scene_position_policy is not None
    OmegaConf.save(raw,args.output/'config.yaml')
    started=time.perf_counter()
    (args.output/'progress.json').write_text(json.dumps(dict(report,stage='native_model_initialization'),indent=2)+'\n')
    video_pipeline=None;pipeline_profile_active=False;source_pin_lease=None;pin_delegate=None;layer_role_probe=None;resident_history=None;generation_profile_active=False;numeric_witness=None;inplace_cache=None;causal_blocks=None
    try:
        def architecture(path,**kwargs):
            cfg=json.loads((Path(path)/'config.json').read_text())
            return CausalWanModel.from_config(cfg,**kwargs)
        from adapters.longlive_sparse.strict_checkpoint_init import StrictCheckpointParameterInit
        with StrictCheckpointParameterInit(enabled=args.constructor_mode=='strict_checkpoint_no_parameter_init') as initialization:
            with patch.object(CausalWanModel,'from_pretrained',side_effect=architecture):
                pipe=CausalDiffusionInferencePipeline(config,device=torch.device('cuda'))
            # Constructor places native T5 on GPU in FP32. Cast before any inference.
            pipe.text_encoder.to(dtype=torch.bfloat16)
            loaded=load_generator_checkpoint(pipe.generator,str(args.assets/'checkpoints/model_bf16.pt'),strict=True)
            if loaded.missing_keys or loaded.unexpected_keys:raise RuntimeError('incomplete released generator')
        report['parameter_initialization_policy']=initialization.record()
        report['strict_generator_load']=dict(missing_keys=loaded.missing_keys,unexpected_keys=loaded.unexpected_keys,
            state_dict_entries=len(pipe.generator.state_dict()))
        report['load_s']=time.perf_counter()-started
        (args.output/'progress.json').write_text(json.dumps(dict(report,stage='native_text_encoding'),indent=2)+'\n')
        text_started=time.perf_counter();encoded={}
        for prompt in dict.fromkeys(prompts[0]):
            encoded[prompt]=pipe.text_encoder(text_prompts=[prompt])['prompt_embeds'].detach().cpu()
        torch.cuda.synchronize();report['native_T5_s']=time.perf_counter()-text_started
        aliases={}
        if 'strip_words' in args.cut_component_ablation:
            from adapters.longlive_sparse.native_cut_ablation import condition_aliases
            aliases=condition_aliases(prompts[0])
            if any(v not in encoded for v in aliases.values()):raise RuntimeError('plain cut text was not independently encoded')
        report['condition_text_aliases']=aliases
        pipe.text_encoder.to('cpu');pipe.text_encoder=CachedNativeTextEncoder(encoded,torch.device('cuda'),aliases)
        gc.collect();torch.cuda.empty_cache()
        pipe.generator.to(device='cuda',dtype=torch.bfloat16).eval().requires_grad_(False)
        if args.pipeline_mode!='none':
            placed=time.perf_counter();pipe.vae.to(device='cuda:1',dtype=torch.bfloat16)
            torch.cuda.synchronize(1);report['pipeline_VAE_placement_s']=time.perf_counter()-placed
        if args.cfg1_positive_cache_only:
            from adapters.longlive_sparse.native_capacity import install_positive_only_allocator
            install_positive_only_allocator(pipe)
        torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        noise=torch.randn(1,length,48,latent_height,latent_width,device='cuda',dtype=torch.bfloat16)
        report['noise_sha256']=tensor_sha256(noise)
        pin_events=[]
        if args.cut_scenario:
            original_pin=pipe._pin_current_chunk
            pin_delegate={'call':original_pin}
            def observe_pin(caches,current_num_frames):
                pin_delegate['call'](caches,current_num_frames)
                pin_events.append(dict(completed_latent=int(caches[0]['global_end_index'])//pipe.frame_seq_length,
                    pinned_start=int(caches[0]['pinned_start']),pinned_tokens=int(caches[0]['pinned_len'])))
            pipe._pin_current_chunk=observe_pin
        replay_log=replay_hook=None;inflight_audit=None;episode_memory=None;causal_scene_memory=None
        if args.native_inplace_cache:
            from adapters.longlive_sparse.native_inplace_cache import NativeInplaceCache
            inplace_cache=NativeInplaceCache(pipe);inplace_cache.attach()
            (args.output/'native_inplace_derived_forward.py').write_text(inplace_cache.derived_source+'\n')
            report.update(causal_model_and_inference_loop_modified=True,
                          native_cache_data_update='inplace_with_original_metadata_commit_order')
        if args.resident_history_policy:
            if (args.episode_memory_mode is not None or args.causal_scene_memory or args.audit_clean_replay
                or args.capture_attention_teacher or args.chest_layer_role_probe or object_state_screen is not None):
                raise ValueError('resident bridge is separate from episode/observer/object protocols')
            from adapters.longlive_sparse.native_resident_history import NativeResidentHistory, NativeResidentConfig
            resident_history=NativeResidentHistory(pipe,NativeResidentConfig(policy=args.resident_history_policy,
                fraction=args.resident_history_fraction,reuse=args.resident_history_reuse))
            resident_history.attach()
            (args.output/'resident_derived_forward.py').write_text(resident_history.derived_source+'\n')
            report.update(causal_model_and_inference_loop_modified=True,
                causal_model_modification='isolated_in_memory_attention_dispatch_only',
                resident_history_config=resident_history.config.__dict__,
                resident_adapter_sha256=hashlib.sha256((ROOT/'adapters/longlive_sparse/native_resident_history.py').read_bytes()).hexdigest())
        if args.causal_block_policy:
            if (not args.native_inplace_cache or args.causal_scene_memory or args.resident_history_policy
                or args.episode_memory_mode is not None or args.audit_clean_replay or args.capture_attention_teacher
                or args.cut_scenario not in ('generated_patchwork_toy_cut_revisit','generated_bead_state_cut_revisit')
                or not args.cfg1_positive_cache_only or args.native_local_frames!=32):
                raise ValueError('source-block memory is its isolated qualified toy/bead native32 protocol')
            from adapters.longlive_sparse.native_causal_block_memory import NativeCausalBlockMemory,CausalBlockConfig
            causal_blocks=NativeCausalBlockMemory(pipe,CausalBlockConfig(policy=args.causal_block_policy,
                fraction=args.causal_block_fraction,grouping=args.causal_block_grouping,head_policy=args.causal_block_heads),
                (latent_height//2,latent_width//2))
            causal_blocks.attach(lambda frame:prompts[0][frame//8])
            (args.output/'causal_block_derived_forward.py').write_text(causal_blocks.derived_source+'\n')
            report.update(causal_model_and_inference_loop_modified=True,causal_block_config=causal_blocks.config.__dict__)
        if args.causal_scene_memory:
            if args.causal_scene_position_policy is None:
                from adapters.longlive_sparse.native_causal_scene_memory import NativeCausalSceneMemory
                causal_scene_memory=NativeCausalSceneMemory(pipe)
            else:
                from adapters.longlive_sparse.native_causal_position_control import NativeCausalPositionControl
                causal_scene_memory=NativeCausalPositionControl(pipe,position_policy=args.causal_scene_position_policy)
            # The producer gets only the current call's string; choose_scene has
            # neither this callback nor the driver's preencoded future dictionary.
            causal_scene_memory.attach(lambda frame:prompts[0][frame//8])
        if args.chest_source_pin_lease:
            from adapters.longlive_sparse.native_source_pin_lease import NativeSourcePinLease
            source_pin_lease=NativeSourcePinLease(pipe,causal_scene_memory)
            pin_delegate['call']=lambda caches,count:source_pin_lease.pin(original_pin,caches,count)
            source_pin_lease.attach()
        if args.episode_memory_mode is not None:
            from adapters.longlive_sparse.native_episode_memory import NativeEpisodeMemory
            episode_type=NativeEpisodeMemory
            episode_kwargs={}
            if args.episode_position_policy!='original':
                from adapters.longlive_sparse.native_retimed_memory import NativeRetimedEpisodeMemory
                episode_type=NativeRetimedEpisodeMemory
                episode_kwargs['position_policy']=args.episode_position_policy
            if args.scene_context_reset:
                from adapters.longlive_sparse.native_scene_context import NativeSceneContextReset
                episode_type=NativeSceneContextReset
                episode_kwargs['anchor_policy']=args.initial_anchor_policy
                if args.memory_reconstruction!='none':
                    from adapters.longlive_sparse.native_semantic_remat import NativeSemanticRematMemory
                    episode_type=NativeSemanticRematMemory
                    episode_kwargs['reconstruction_condition']=args.memory_reconstruction
            source_end,target_start=episode_source_and_target(segments)
            if reviewed_protocol is not None:
                if (source_end,target_start)!=(reviewed_protocol['spec']['source_end'],reviewed_protocol['spec']['target_start']):
                    raise ValueError('reviewed source boundaries changed')
            episode_memory=episode_type(pipe,mode=args.episode_memory_mode,
                source_end=source_end,target_start=target_start,prompts=prompts[0],
                destination=args.episode_destination,restore_after_frames=args.episode_restore_after_frames,**episode_kwargs)
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
        if args.chest_layer_role_probe:
            from adapters.longlive_sparse.native_layer_role_probe import NativeLayerRoleProbe
            layer_role_probe=NativeLayerRoleProbe(pipe,token_grid=(latent_height//2,latent_width//2))
            layer_role_probe.attach()
        if args.native_numeric_witness:
            if (not args.chest_hybrid_study or not args.equivalence_reference or args.seed!=20260925
                or args.chest_layer_role_probe or args.capture_attention_teacher or args.resident_history_policy):
                raise ValueError('minimal numeric witness is the isolated existing seed25 hybrid only')
            from adapters.longlive_sparse.native_numeric_witness import NativeNumericWitness
            numeric_witness=NativeNumericWitness(pipe,(latent_height//2,latent_width//2));numeric_witness.attach()
        attention_teacher=None
        if args.capture_attention_teacher:
            if not args.cut_scenario or args.audit_clean_replay or args.episode_memory_mode=='log_reveal':
                raise ValueError('attention teacher is an isolated raw/native cut capture')
            from adapters.longlive_sparse.native_attention_teacher import NativeAttentionTeacherCapture
            attention_teacher=NativeAttentionTeacherCapture(pipe,query_frame=segments[-1]['start_latent'],
                token_grid=(latent_height//2,latent_width//2))
            attention_teacher.attach()
        rope_freeze=None
        if 'freeze_rope' in args.cut_component_ablation:
            from adapters.longlive_sparse.native_cut_ablation import NativeRopePhaseFreeze
            rope_freeze=NativeRopePhaseFreeze(pipe);rope_freeze.attach()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();generation_started=time.perf_counter()
        if args.pipeline_profile:
            torch.cuda.profiler.start();torch.cuda.nvtx.range_push('native_pipeline/full_run');pipeline_profile_active=True
        if args.pipeline_mode!='none':
            from adapters.longlive_sparse.native_video_pipeline import NativeVideoPipeline
            from wan_5b.modules.vae2_2 import unpatchify
            torch.cuda.reset_peak_memory_stats(1)
            pipeline_sink=IncrementalVideoSink(args.output/'video.mp4',expected_frames=4*length-3,started=generation_started,fps=24)
            video_pipeline=NativeVideoPipeline(pipe.vae,unpatchify,pipeline_sink,source_device='cuda:0',target_device='cuda:1',
                latent_shape=report['latent_shape'],started=generation_started,slots=args.pipeline_slots,
                pinned_budget=args.pipeline_pinned_mib*1024**2,serial=args.pipeline_mode=='serial',
                encode_mode=args.pipeline_encode_mode,pixel_slots=args.pipeline_pixel_slots)
            video_pipeline.attach(pipe)
        (args.output/'progress.json').write_text(json.dumps(dict(report,stage='native_generation'),indent=2)+'\n')
        if args.generation_profile:
            if args.pipeline_profile:raise ValueError('choose one profiling window')
            torch.cuda.profiler.start();torch.cuda.nvtx.range_push('native_generation_only');generation_profile_active=True
        latent=pipe.inference(noise=noise,text_prompts=prompts,return_latents=True)
        torch.cuda.synchronize();report['native_DiT_s']=time.perf_counter()-generation_started
        if generation_profile_active:
            torch.cuda.nvtx.range_pop();torch.cuda.profiler.stop();generation_profile_active=False
            report['generation_profile']='cudaProfilerApi_native_generation_only_not_unprofiled_timing'
        if video_pipeline is not None:
            video_pipeline.detach();report['native_DiT_s_includes_pipeline_backpressure']=True
        if rope_freeze is not None:
            rope_freeze.detach();report['rope_phase_ablation']=rope_freeze.audit()
        # Reverse attachment order: the teacher wrapper surrounds the episode's
        # shape observer. Neither wrapper may survive into subsequent use.
        if attention_teacher is not None:
            attention_teacher.detach()
        if layer_role_probe is not None:layer_role_probe.detach()
        if numeric_witness is not None:numeric_witness.detach()
        if resident_history is not None:
            resident_history.detach();report['resident_history']=resident_history.audit()
        if causal_blocks is not None:
            causal_blocks.detach();report['causal_block_memory']=causal_blocks.audit()
        if inplace_cache is not None:
            inplace_cache.detach();report['native_inplace_cache']=inplace_cache.audit()
        if episode_memory is not None:
            episode_memory.detach();report['episode_memory']=episode_memory.audit()
        if causal_scene_memory is not None:
            causal_scene_memory.detach();report['causal_scene_memory']=causal_scene_memory.audit()
        if video_pipeline is not None:
            # Final delivery is timed before offline hashing and artifact writes.
            report['pixels'],report['video_pipeline']=video_pipeline.finish(generation_finished_s=report['native_DiT_s'])
            if pipeline_profile_active:
                torch.cuda.nvtx.range_pop();torch.cuda.profiler.stop();pipeline_profile_active=False
            video_pipeline.write_trace(args.output/'pipeline_host_trace.json')
            report['pipeline_VAE_GPU_peak_allocated_bytes']=torch.cuda.max_memory_allocated(1)
        if args.cut_scenario:
            report['pre_return_latent_sha256']=tensor_sha256(latent[:,:segments[-1]['start_latent']])
            report['first_return_latent_sha256']=tensor_sha256(latent[:,segments[-1]['start_latent']:segments[-1]['start_latent']+8])
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
        if video_pipeline is not None:
            if report['video_pipeline']['streamed_latent_sha256']!=report['latent_sha256']:
                raise RuntimeError('streamed committed latents differ from final native output')
        else:
            offload_started=time.perf_counter()
            pipe.kv_cache_pos=pipe.kv_cache_neg=pipe.crossattn_cache_pos=pipe.crossattn_cache_neg=None
            pipe.generator.to('cpu');gc.collect();torch.cuda.empty_cache()
            pipe.vae.to(device='cuda',dtype=torch.bfloat16)
            torch.cuda.synchronize();report['generation_to_decode_placement_s']=time.perf_counter()-offload_started
            decode_started=time.perf_counter();video=pipe.vae.decode_to_pixel(latent)
            torch.cuda.synchronize();report['native_VAE_s']=time.perf_counter()-decode_started
            sink=IncrementalVideoSink(args.output/'video.mp4',expected_frames=4*length-3,started=generation_started,fps=24)
            sink(video);report['pixels']=sink.close()
        # Preserve complete generation artifacts even if diagnostic export fails.
        if attention_teacher is not None:
            report['attention_teacher']=attention_teacher.export(args.output/'attention_teacher.pt')
        if layer_role_probe is not None:report['layer_role_probe']=layer_role_probe.export(args.output/'layer_role_probe.pt')
        if numeric_witness is not None:report['numeric_witness']=numeric_witness.export(args.output/'numeric_witness.pt')
        if causal_blocks is not None:report['causal_block_routes']=causal_blocks.export_routes(args.output/'causal_block_routes.pt')
        if external is not None:
            for key in ('noise_sha256','latent_sha256'):
                if report[key]!=external[key]:raise RuntimeError('observer changed generated trajectory')
            if report['pixels']['raw_RGB_sha256']!=external['pixels']['raw_RGB_sha256']:raise RuntimeError('observer/replay changed decoded RGB')
            report['observer_noise_latent_RGB_equivalence']=True
        report.update(status='pass',video=str(args.output/'video.mp4'),wall_including_loading_s=time.perf_counter()-started,
            quality='pending_own_identity_absence_and_transition_review')
    except BaseException:
        report.update(status='fail',traceback=traceback.format_exc(),partial_artifacts_preserved=True)
        if video_pipeline is not None:
            try:
                video_pipeline.abort()
                if not (args.output/'pipeline_host_trace.json').exists():video_pipeline.write_trace(args.output/'pipeline_host_trace.json')
            except BaseException:report['pipeline_cleanup_traceback']=traceback.format_exc()
        if pipeline_profile_active:
            try:torch.cuda.nvtx.range_pop();torch.cuda.profiler.stop()
            except BaseException:report['pipeline_profile_cleanup_traceback']=traceback.format_exc()
        raise
    finally:
        if numeric_witness is not None:
            numeric_witness.detach()
            if numeric_witness.records and not (args.output/'numeric_witness.pt').exists():
                report['numeric_witness']=numeric_witness.export(args.output/'numeric_witness.pt',allow_partial=True)
        if generation_profile_active:
            torch.cuda.nvtx.range_pop();torch.cuda.profiler.stop()
        if resident_history is not None:
            resident_history.detach()
            if hasattr(resident_history,'derived_sha256'):
                report['resident_history']=resident_history.audit()
        if causal_blocks is not None:
            causal_blocks.detach()
            if hasattr(causal_blocks,'derived_sha256'):
                report['causal_block_memory']=causal_blocks.audit()
                if not (args.output/'causal_block_routes.pt').exists():
                    report['causal_block_routes']=causal_blocks.export_routes(args.output/'causal_block_routes.pt')
        if inplace_cache is not None:
            inplace_cache.detach()
            if hasattr(inplace_cache,'derived_sha256'):report['native_inplace_cache']=inplace_cache.audit()
        if layer_role_probe is not None:
            layer_role_probe.detach()
            if report.get('status')!='pass' and layer_role_probe.records and not (args.output/'layer_role_probe.pt').exists():
                report['layer_role_probe']=layer_role_probe.export(args.output/'layer_role_probe.pt',allow_partial=True)
        if source_pin_lease is not None:
            source_pin_lease.detach();report['source_pin_lease']=source_pin_lease.audit()
            if pin_delegate is not None:pin_delegate['call']=original_pin
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('segments','source_files_sha256','traceback','resident_history','causal_block_memory')}),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException:
        if CREATED_OUTPUT is not None and not (CREATED_OUTPUT/'summary.json').exists():
            with (CREATED_OUTPUT/'summary.json').open('x') as handle:
                json.dump(dict(status='fail',stage='preflight_or_import',traceback=traceback.format_exc(),
                    partial_artifacts_preserved=True),handle,indent=2);handle.write('\n')
        raise
