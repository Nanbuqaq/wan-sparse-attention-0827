"""Frozen new-state feasibility data, not a new memory algorithm."""
import hashlib
import json


SCENARIOS=('chest_revisit','chest_visible_control','envelope_revisit','envelope_visible_control')


def expand_object_scenario(spec,scenario):
    choice=next(s for s in spec['scenarios'] if s['id']==scenario)
    obj=next(o for o in spec['objects'] if o['id']==choice['object'])
    visible=choice['mode']=='visible'
    return dict(id=scenario,segments=[
        dict(start_latent=0,role='initial_state',scene_cut=False,prompt=obj['initial']),
        dict(start_latent=16,role='state_transition',scene_cut=True,prompt=obj['transition']),
        dict(start_latent=32,role='settled_source',scene_cut=False,prompt=obj['settled']),
        dict(start_latent=48,role='visible_control' if visible else 'away',scene_cut=True,
             prompt=obj['visible'] if visible else spec['away_prompt']),
        dict(start_latent=96,role='visible_control_return_cut' if visible else 'return_without_restatement',scene_cut=True,
             prompt=obj['visible'] if visible else obj['return']),
    ])


def validate_object_state_screen(args,root):
    if getattr(args,'cut_scenario',None) not in SCENARIOS:return None
    path=root/'configs/system/native_object_state_screen.json';spec=json.loads(path.read_text())
    required=dict(gate=False,episode_gate_layout=False,native_local_frames=32,cfg1_positive_cache_only=True,
                  fixed_adaln_warps=16,fixed_adaln_stages=1,episode_memory_mode=None,causal_scene_memory=False,
                  memory_reconstruction='none',cut_component_ablation='none',initial_anchor_policy='keep',
                  scene_context_reset=False,capture_attention_teacher=False,audit_clean_replay=False,
                  replay_resume_after_latents=0,pipeline_mode='none',pipeline_profile=False,reviewed_memory_protocol=None,control=None)
    for key,value in required.items():
        if getattr(args,key,value)!=value:raise ValueError(f'new object-state screen is Dense-only; invalid {key}')
    if args.seed not in spec['seeds'] or not spec['screen_only']:
        raise ValueError('new object-state screen seed/protocol is not frozen')
    return dict(config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scope=spec['scope'],
                formal_holdout=False,source_validity_required_before_memory=True,seed=args.seed)
