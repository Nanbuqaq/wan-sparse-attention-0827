"""Fixed native feasibility protocols; metadata is never a routing input."""
import json

STATE_SCENARIOS=tuple(f'w2_state_{task}_{operation}' for task in ('blue_box','silver_case') for operation in ('keep','close'))


def state_update_schedule(root,scenario,*,gate=False,episode_gate=False):
    if scenario not in STATE_SCENARIOS:raise ValueError('unknown frozen state protocol')
    if gate and not episode_gate:raise ValueError('state smoke uses native64 gate layout')
    task,operation=scenario.removeprefix('w2_state_').rsplit('_',1)
    spec=json.loads((root/'configs/system/state_update_protocols.json').read_text());p=spec['protocols'][task]
    starts=(0,8,16,48) if gate else (0,8,48,96)
    texts=(spec['initial_prompt'],p['source_prompt'],spec['away_prompt'],p['return_'+operation])
    roles=('initial_target_free','open_source_requested','away','return_without_restatement')
    segments=[dict(start_latent=start,scene_cut=i>0,prompt=text,role=role)
              for i,(start,text,role) in enumerate(zip(starts,texts,roles))]
    length=64 if gate else spec['latent_frames'];prompts=[]
    for frame in range(0,length,8):
        s=next(s for s in reversed(segments) if s['start_latent']<=frame)
        prompts.append(('The scene transitions. ' if s['scene_cut'] and frame==s['start_latent'] else '')+s['prompt'])
    return segments,[prompts]
