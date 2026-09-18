"""Fixed native feasibility protocols; metadata is never a routing input."""
import json

STATE_SCENARIOS=tuple(f'w2_state_{task}_{operation}' for task in ('blue_box','silver_case','red_toolbox') for operation in ('keep','close'))
STATE_SCENARIOS+=('w2_state_silver_case_last_state','w2_state_red_toolbox_last_state','w2_state_red_toolbox_last_open')
STATE_SCENARIOS+=('w2_state_pattern_tile_keep',)
# Same-object two-state interaction diagnostics: identical closed->open history as
# w2_state_red_toolbox_last_open, only the return request differs (keep-open vs reclose).
STATE_SCENARIOS+=('w2_state_red_toolbox_hold_open','w2_state_red_toolbox_reclose')


def state_update_schedule(root,scenario,*,gate=False,episode_gate=False):
    if scenario not in STATE_SCENARIOS:raise ValueError('unknown frozen state protocol')
    if gate and not episode_gate:raise ValueError('state smoke uses native64 gate layout')
    if scenario in ('w2_state_red_toolbox_hold_open','w2_state_red_toolbox_reclose'):
        return state_two_state_return_schedule(root,gate=gate,keep_open=scenario.endswith('_hold_open'))
    if scenario in ('w2_state_silver_case_last_state','w2_state_red_toolbox_last_state','w2_state_red_toolbox_last_open'):
        task='red_toolbox' if 'red_toolbox' in scenario else 'silver_case'
        return state_last_configuration_schedule(root,gate=gate,task=task,update='open' if scenario.endswith('_last_open') else 'close')
    task,operation=scenario.removeprefix('w2_state_').rsplit('_',1)
    spec=json.loads((root/'configs/system/state_update_protocols.json').read_text());p=spec['protocols'][task]
    starts=(0,8,16,48) if gate else (0,8,48,96)
    texts=(spec['initial_prompt'],p['source_prompt'],spec['away_prompt'],p['return_'+operation])
    roles=('initial_target_free',p.get('source_role','open_source_requested'),'away','return_without_restatement')
    segments=[dict(start_latent=start,scene_cut=i>0,prompt=text,role=role)
              for i,(start,text,role) in enumerate(zip(starts,texts,roles))]
    length=64 if gate else spec['latent_frames'];prompts=[]
    for frame in range(0,length,8):
        s=next(s for s in reversed(segments) if s['start_latent']<=frame)
        prompts.append(('The scene transitions. ' if s['scene_cut'] and frame==s['start_latent'] else '')+s['prompt'])
    return segments,[prompts]


def state_last_configuration_schedule(root,*,gate=False,task='silver_case',update='close'):
    """A real update within one phase; return does not restate the target state."""
    spec=json.loads((root/'configs/system/state_update_protocols.json').read_text())
    if (task,update) not in (('silver_case','close'),('red_toolbox','close'),('red_toolbox','open')):
        raise ValueError('unregistered source update direction')
    p=spec['protocols'][task]
    starts=(0,8,16,32,48) if gate else (0,8,24,48,96)
    texts=(spec['initial_prompt'],p['source_closed_prompt'] if update=='open' else p['source_prompt'],p['update_'+update],spec['away_prompt'],p['return_last_state'])
    roles=('initial_target_free','closed_requested' if update=='open' else 'open_requested',update+'_requested_same_scene','away','latest_state_not_restated')
    cuts=(False,True,False,True,True)
    segments=[dict(start_latent=start,scene_cut=cut,prompt=text,role=role)
        for start,text,role,cut in zip(starts,texts,roles,cuts)]
    length=64 if gate else spec['latent_frames']
    prompts=[]
    for frame in range(0,length,8):
        s=next(s for s in reversed(segments) if s['start_latent']<=frame)
        prompts.append(('The scene transitions. ' if s['scene_cut'] and frame==s['start_latent'] else '')+s['prompt'])
    return segments,[prompts]


def state_two_state_return_schedule(root,*,gate=False,keep_open=True):
    """closed->open history (identical to last_open source) + an explicit return request.

    The two return requests share the SAME valid closed->open prefix so that only the
    current request differs: keep_open = preserve the observed open state
    (return_keep), reclose = shut the lid again (return_close). Both restate the
    target state explicitly, unlike last_open's return_last_state.
    """
    spec=json.loads((root/'configs/system/state_update_protocols.json').read_text())
    p=spec['protocols']['red_toolbox']
    starts=(0,8,16,32,48) if gate else (0,8,24,48,96)
    return_prompt=p['return_keep'] if keep_open else p['return_close']
    return_role='return_keep_open' if keep_open else 'return_reclose'
    texts=(spec['initial_prompt'],p['source_closed_prompt'],p['update_open'],spec['away_prompt'],return_prompt)
    roles=('initial_target_free','closed_requested','open_requested_same_scene','away',return_role)
    cuts=(False,True,False,True,True)
    segments=[dict(start_latent=start,scene_cut=cut,prompt=text,role=role)
        for start,text,role,cut in zip(starts,texts,roles,cuts)]
    length=64 if gate else spec['latent_frames']
    prompts=[]
    for frame in range(0,length,8):
        s=next(s for s in reversed(segments) if s['start_latent']<=frame)
        prompts.append(('The scene transitions. ' if s['scene_cut'] and frame==s['start_latent'] else '')+s['prompt'])
    return segments,[prompts]
    return segments,[prompts]
