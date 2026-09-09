#!/usr/bin/env python3
"""Full-resolution runtime gate against a previously generated Dense prefix."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import av
    import torch
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2)
    path=args.case/'summary.json';control_path=args.control/'summary.json'
    d=json.loads(path.read_text());base=json.loads(control_path.read_text())
    assert d['status']==base['status']=='pass'
    for key in ('seed','cut_scenario','prompts_per_block','latent_shape','noise_sha256','pre_return_latent_sha256',
                'source_files_sha256','upstream_source_SHA','assets_manifest_sha256','attention_backend',
                'KV_and_generator_dtype','triton_version','fixed_native_adaln_recipe','native_shot_pin_events'):
        assert d[key]==base[key],key
    assert d['latent_shape']==[1,128,48,44,80] and d['cut_scenario']=='settled_bead_revisit'
    assert d['reviewed_memory_protocol']['spec']['id']=='settled_state_v1'
    assert d['episode_memory_mode']=='raw_reveal' and base['episode_memory_mode'] is None
    m=d['episode_memory'];plan=m['installation']['admission_plan']
    assert plan['source_frames']==list(range(40,48)) and plan['target_start']==96
    assert m['capture']['source_temporal_offset']==8 and m['capture']['completed_latents']==48
    assert plan['destination_token_range']==[7040,14080] and m['installation']['cache_metadata_unchanged']
    assert m['ledger']['demand_H2D_payload_bytes']==m['ledger']['archive_D2H_payload_bytes']==2595225600
    assert m['ledger']['demand_D2D_KV_bytes']==0 and d['local_frames']==32
    if d['episode_position_policy']=='recent_virtual':
        assert plan['temporal_delta']==64 and plan['virtual_source_frames']==list(range(88,96))
    else:assert d['episode_position_policy']=='original'
    latent=torch.load(args.case/'latents.pt',map_location='cpu',weights_only=True)
    reference=torch.load(args.control/'latents.pt',map_location='cpu',weights_only=True)
    assert torch.equal(latent[:,:96],reference[:,:96])
    hashes=[]
    for root in (args.case,args.control):
        digest=hashlib.sha256();count=0
        with av.open(str(root/'video.mp4')) as container:
            for i,frame in enumerate(container.decode(video=0)):
                if i<381:digest.update(frame.to_ndarray(format='rgb24').tobytes())
                count+=1
        assert count==509;hashes.append(digest.hexdigest())
    assert hashes[0]==hashes[1]
    report=dict(status='pass',seed=d['seed'],policy=d['episode_position_policy'],actual_pre96_latent_exact=True,
        actual_prefix_decoded_RGB_sha256=hashes[0],source_frames=plan['source_frames'],ledger=m['ledger'],
        summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        control_summary_sha256=hashlib.sha256(control_path.read_bytes()).hexdigest(),
        quality_not_evaluated_by_runtime_gate=True)
    with args.output.open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps(dict(status='pass',seed=d['seed'],policy=d['episode_position_policy'])))


if __name__=='__main__':main()
