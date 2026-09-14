"""Two fixed multi-event write controls: capacity at B return, version at A return."""
import json

WRITE_SCENARIOS=('w2_write_return_b','w2_write_return_a')


def write_origin_schedule(root,scenario):
    if scenario not in WRITE_SCENARIOS:raise ValueError('unknown frozen write-origin protocol')
    spec=json.loads((root/'configs/system/access_multi_event.json').read_text())
    segments=[dict(s) for s in spec['segments']]
    if scenario=='w2_write_return_a':
        segments[-1]['prompt']=segments[3]['prompt']
        segments[-1]['role']='A_second_return'
    prompts=[]
    for frame in range(0,spec['latent_frames'],8):
        s=next(s for s in reversed(segments) if s['start_latent']<=frame)
        prompts.append(('The scene transitions. ' if s['scene_cut'] and frame==s['start_latent'] else '')+s['prompt'])
    return segments,[prompts]
