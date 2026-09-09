#!/usr/bin/env python3
"""Offline verification only; expected source bounds never enter the selector."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import av
    import torch
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2)
    path=args.case/'summary.json';refpath=args.reference/'summary.json'
    d=json.loads(path.read_text());ref=json.loads(refpath.read_text())
    assert d['status']==ref['status']=='pass' and d['observer_noise_latent_RGB_equivalence']
    for key in ('noise_sha256','latent_sha256','latent_shape','prompts_per_block','native_shot_pin_events','fixed_native_adaln_recipe','triton_version'):
        assert d[key]==ref[key],key
    assert d['latent_shape']==[1,64,48,16,32]
    m=d['causal_scene_memory'];assert len(m['archives'])==3 and len(m['installations'])==1
    assert [r['source_end'] for r in m['archives']]==[8,16,48]
    install=m['installations'][0];plan=install['installation']['admission_plan']
    assert install['at_latent']==48 and plan['source_frames']==list(range(8,16)) and plan['temporal_delta']==48
    assert plan['method']=='causal_cue_T5_latest_scene_history'
    assert m['ledger']['history_H2D_KV_bytes']==377487360 and m['ledger']['archive_D2H_KV_bytes']==3*377487360
    assert m['ledger']['CPU_archive_peak_tensor_bytes']<=m['archive_budget_bytes']
    assert not m['raw_source_frames_or_target_frames_supplied_to_selector']
    assert torch.equal(torch.load(args.case/'latents.pt',map_location='cpu',weights_only=True),torch.load(args.reference/'latents.pt',map_location='cpu',weights_only=True))
    hashes=[]
    for root in (args.case,args.reference):
        digest=hashlib.sha256();count=0
        with av.open(str(root/'video.mp4')) as container:
            for frame in container.decode(video=0):digest.update(frame.to_ndarray(format='rgb24').tobytes());count+=1
        assert count==253;hashes.append(digest.hexdigest())
    assert hashes[0]==hashes[1]
    report=dict(status='pass',full_actual_latent_and_decoded_RGB_exact=True,decoded_RGB_sha256=hashes[0],
        causal_source_and_target_match_offline_reference=True,ledger=m['ledger'],
        case_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference_summary_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),
        quality_not_evaluated_on_low_resolution_gate=True,not_a_general_entity_tracker=True)
    with args.output.open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps(dict(status='pass',full_actual_latent_and_decoded_RGB_exact=True)))


if __name__=='__main__':main()
