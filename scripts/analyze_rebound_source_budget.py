#!/usr/bin/env python3
"""Offline same-budget source retention on actual rebound Attention captures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_native_state_region import partition_record


def nearby_nonred(pile,all_red,*,frames=8,height=22,width=40):
    pile=np.asarray(pile,dtype=bool).reshape(frames,height,width)
    red=np.asarray(all_red,dtype=bool).reshape(frames,height,width);selected=[]
    for frame in range(frames):
        source=np.argwhere(pile[frame]);background=np.argwhere(~red[frame]);n=len(source)
        if not n or len(background)<n:raise ValueError('cannot build equal-count nearby nonred control')
        distances=((background[:,None,:]-source[None,:,:])**2).sum(-1).min(-1)
        chosen=background[np.argsort(distances,kind='stable')[:n]]
        selected.extend((frame*height*width+y*width+x) for y,x in chosen)
    return sorted(map(int,selected))


def energy_indices(value,counts,frame_tokens=880):
    energy=value.float().square().sum(-1).mean(-1).reshape(len(counts),frame_tokens)
    return sorted(frame*frame_tokens+int(i) for frame,n in enumerate(counts)
                  for i in torch.argsort(energy[frame],descending=True,stable=True)[:int(n)])


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--region',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    summary=json.loads((args.capture/'summary.json').read_text());assert summary['status']=='pass' and summary['observer_noise_latent_RGB_equivalence']
    region=json.loads(args.region.read_text());source_root=Path(region['source_video']).parent
    source_summary=json.loads((source_root/'summary.json').read_text())
    assert source_summary['pre_return_latent_sha256']==summary['pre_return_latent_sha256']
    assert torch.equal(torch.load(source_root/'latents.pt',map_location='cpu',weights_only=True)[:,:96],
                       torch.load(args.capture/'latents.pt',map_location='cpu',weights_only=True)[:,:96])
    path=args.capture/'attention_teacher.pt'
    with path.open('rb') as handle:capture_sha=hashlib.file_digest(handle,'sha256').hexdigest()
    assert capture_sha==summary['attention_teacher']['sha256']
    records=torch.load(path,map_location='cpu',weights_only=True)['records']
    red=[i-7040 for i in region['partitions']['source_red_pile']]
    all_red=red+[i-7040 for i in region['partitions']['source_other_red']]
    red_mask=np.zeros(7040,dtype=bool);red_mask[red]=True
    any_red=np.zeros(7040,dtype=bool);any_red[all_red]=True
    counts=red_mask.reshape(8,880).sum(-1).tolist()
    fixed=dict(red_proxy=red,random_nonred=[i-7040 for i in region['partitions']['matched_nonred']],
               nearby_nonred=nearby_nonred(red_mask,any_red))
    masks={};source_values={};rows=[]
    for record in sorted(records,key=lambda r:(r['phase'],r['layer'])):
        layer=record['layer'];source_value=record['v'][0,7040:14080]
        if layer not in masks:
            source_values[layer]=source_value
            masks[layer]=dict(fixed,V_energy_matched=energy_indices(source_value,counts),
                             V_energy_quarter=energy_indices(source_value,[220]*8))
        assert torch.equal(source_value,source_values[layer])
        for name,keep in masks[layer].items():
            keep=sorted(set(keep));drop=sorted(set(range(7040))-set(keep))
            assert record['k'].shape[1]==28160
            parts=dict(exact_context=list(range(7040))+list(range(14080,28160)),
                       kept_source=[i+7040 for i in keep],dropped_source=[i+7040 for i in drop])
            analysis=partition_record(record,parts);assert analysis['FP32_gate']
            rows.append(dict(phase=record['phase'],layer=layer,policy=name,kept_source_tokens=len(keep),source_fraction=len(keep)/7040,
                selected_red_overlap=len(set(keep)&set(red)),reference_FP32_error=analysis['FP32_error'],
                sparse_output_change_per_query=analysis['parts']['dropped_source']['remove_output_sensitivity'],
                lower_center_output_change=analysis['parts']['dropped_source']['lower_center_remove_sensitivity'],
                kept_source_probability_mass=analysis['parts']['kept_source']['mass']))
        print(json.dumps(dict(phase=record['phase'],layer=layer,policies=len(masks[layer]))),flush=True)
    report=dict(status='pass',rows=rows,source_masks_by_layer={str(k):v for k,v in masks.items()},
        capture_sha256=capture_sha,region_sha256=hashlib.sha256(args.region.read_bytes()).hexdigest(),
        actual_source_prefix_latents_equal=True,all_full_reference_FP32_gates=True,
        tier_notes='first four policies use equal source tokens and per-frame counts; quarter energy uses larger 25% budget and is separate',
        limitations=['fixed captured Q, not closed-loop video quality','32 geometric queries per head, not full Q average',
            'red proxy includes source motion and is not a semantic mask','nearby nonred matches count/frame and proximity, not all spatial confounds',
            'V-energy is archive-computable, but its online video implementation is not present',
            'no byte or speed claim from output error; physical layout costs have separate measurements'])
    (args.output/'source_budget_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='pass',records=9,mask_evaluations=len(rows),video_quality_evaluated=False)))


if __name__=='__main__':main()
