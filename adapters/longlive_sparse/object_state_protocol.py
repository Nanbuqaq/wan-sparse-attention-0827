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
    study=bool(getattr(args,'object_state_memory_study',False));registration=None
    text_control=getattr(args,'object_state_text_control',None);text_registration=None
    hybrid=bool(getattr(args,'chest_hybrid_study',False));hybrid_registration=None
    lease=bool(getattr(args,'chest_source_pin_lease',False));lease_registration=None
    probe=bool(getattr(args,'chest_layer_role_probe',False));probe_registration=None
    if probe:
        probe_path=root/'configs/system/native_chest_layer_role_probe.json';probe_registration=json.loads(probe_path.read_text())
        if (not hybrid or lease or args.cut_scenario not in probe_registration['scenarios'] or args.seed not in probe_registration['seeds']
                or probe_registration['hybrid_registration_sha256']!=hashlib.sha256((root/'configs/system/native_chest_hybrid_control.json').read_bytes()).hexdigest()):
            raise ValueError('unregistered or mixed all-layer role probe')
    if lease:
        lease_path=root/'configs/system/native_chest_source_pin_lease.json';lease_registration=json.loads(lease_path.read_text())
        if (not hybrid or args.cut_scenario not in lease_registration['scenarios'] or args.seed not in lease_registration['seeds']
                or lease_registration['hybrid_registration_sha256']!=hashlib.sha256((root/'configs/system/native_chest_hybrid_control.json').read_bytes()).hexdigest()):
            raise ValueError('unregistered source-pin lifetime diagnostic')
    if hybrid:
        hybrid_path=root/'configs/system/native_chest_hybrid_control.json';hybrid_registration=json.loads(hybrid_path.read_text())
        if (not study or text_control!='past_settled_restatement'
                or args.cut_scenario not in hybrid_registration['scenarios'] or args.seed not in hybrid_registration['seeds']
                or getattr(args,'causal_scene_position_policy',None)!=hybrid_registration['position_policy']):
            raise ValueError('unregistered chest condition/history factorial cell')
        for name,filename in [('memory_registration_sha256','native_object_state_memory.json'),
                              ('text_registration_sha256','native_chest_text_control.json')]:
            if hybrid_registration[name]!=hashlib.sha256((root/'configs/system'/filename).read_bytes()).hexdigest():
                raise ValueError('hybrid parent registration changed')
    if text_control:
        text_path=root/'configs/system/native_chest_text_control.json';text_registration=json.loads(text_path.read_text())
        if ((study and not hybrid) or text_control!=text_registration['id'] or args.cut_scenario not in text_registration['scenarios']
                or args.seed not in text_registration['seeds']
                or text_registration['screen_config_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError('unregistered or mixed object-state text control')
    if study:
        registration_path=root/'configs/system/native_object_state_memory.json'
        registration=json.loads(registration_path.read_text())
        if (args.cut_scenario not in registration['scenarios'] or args.seed not in registration['seeds']
                or registration['screen_config_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest()
                or getattr(args,'causal_scene_position_policy','recent_virtual') not in registration['position_policies']):
            raise ValueError('object-state memory study is not registered for this case')
    required=dict(gate=False,episode_gate_layout=False,native_local_frames=32,cfg1_positive_cache_only=True,constructor_mode='reference',
                  fixed_adaln_warps=16,fixed_adaln_stages=1,episode_memory_mode=None,causal_scene_memory=study,
                  memory_reconstruction='none',cut_component_ablation='none',initial_anchor_policy='keep',
                  scene_context_reset=False,capture_attention_teacher=False,audit_clean_replay=False,
                  replay_resume_after_latents=0,pipeline_mode='none',pipeline_encode_mode='inline',pipeline_profile=False,reviewed_memory_protocol=None,control=None)
    for key,value in required.items():
        if getattr(args,key,value)!=value:raise ValueError(f'object-state protocol forbids {key}')
    if args.seed not in spec['seeds'] or not spec['screen_only']:
        raise ValueError('new object-state screen seed/protocol is not frozen')
    return dict(config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope=probe_registration['scope'] if probe else lease_registration['scope'] if lease else hybrid_registration['scope'] if hybrid else registration['scope'] if study else text_registration['scope'] if text_control else spec['scope'],
                Dense_only=not study,formal_holdout=False,source_validity_required_before_memory=True,seed=args.seed,
                memory_study_registration=registration,
                memory_registration_sha256=hashlib.sha256(registration_path.read_bytes()).hexdigest() if study else None,
                text_control=text_control,text_control_registration=text_registration,
                text_control_registration_sha256=hashlib.sha256(text_path.read_bytes()).hexdigest() if text_control else None,
                hybrid_registration=hybrid_registration,
                hybrid_registration_sha256=hashlib.sha256(hybrid_path.read_bytes()).hexdigest() if hybrid else None,
                source_pin_lease_registration=lease_registration,
                source_pin_lease_registration_sha256=hashlib.sha256(lease_path.read_bytes()).hexdigest() if lease else None,
                layer_role_probe_registration=probe_registration,
                layer_role_probe_registration_sha256=hashlib.sha256(probe_path.read_bytes()).hexdigest() if probe else None)


def apply_past_text_control(selected,registration):
    segments=[dict(s) for s in selected['segments']]
    sources=[s for s in segments if s['role']==registration['source_role']]
    targets=[s for s in segments if s['role']==registration['target_role']]
    if len(sources)!=1 or len(targets)!=1 or sources[0]['start_latent']>=targets[0]['start_latent']:
        raise ValueError('one strictly past source and one return required')
    targets[0]['prompt']+=' '+sources[0]['prompt']
    targets[0]['role']='return_with_past_text_restatement'
    return dict(selected,segments=segments)
