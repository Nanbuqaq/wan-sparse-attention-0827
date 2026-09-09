#!/usr/bin/env python3
"""Offline past-red-pixel proxy and equal-token local output sensitivity.

These are geometric color proxies, not semantic masks, quality scores, or an
online selector. Q/K/V/O are from the already verified isolated captures.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_native_attention_teacher import output_error,summary


def red_cells(rgb,grid=(22,40)):
    x=rgb.astype(np.float32)/255.;r,g,b=x[...,0],x[...,1],x[...,2]
    high=x.max(-1);low=x.min(-1);delta=high-low;denom=np.maximum(delta,1e-6)
    hue=np.where(high==r,((g-b)/denom)%6,np.where(high==g,(b-r)/denom+2,(r-g)/denom+4))*60
    sat=delta/np.maximum(high,1e-6)
    mask=((hue<=15)|(hue>=330))&(sat>=.5)&(high>=.2)&(delta>0)
    gh,gw=grid;h,w=mask.shape
    if h%gh or w%gw:raise ValueError('source image cannot be aligned to token grid')
    fraction=mask.reshape(gh,h//gh,gw,w//gw).mean((1,3))
    active=fraction>=.05;seen=np.zeros_like(active);components=[]
    for y,x in zip(*np.where(active)):
        if seen[y,x]:continue
        queue=deque([(int(y),int(x))]);seen[y,x]=True;cells=[]
        while queue:
            cy,cx=queue.popleft();cells.append((cy,cx))
            for ny,nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0<=ny<gh and 0<=nx<gw and active[ny,nx] and not seen[ny,nx]:
                    seen[ny,nx]=True;queue.append((ny,nx))
        components.append(cells)
    pile=np.zeros_like(active)
    if components:
        largest=max(components,key=lambda cells:sum(fraction[y,x] for y,x in cells))
        for y,x in largest:pile[y,x]=True
    return fraction,active,pile


@torch.inference_mode()
def partition_record(record,parts):
    q=record['q'][0].permute(1,0,2).float().contiguous();outputs=[];zs=[];names=[]
    keys=[]
    for name,index in parts.items():
        if not len(index):continue
        index=torch.tensor(index,dtype=torch.long);keys.extend(index.tolist());names.append(name)
        k=record['k'][0].index_select(0,index).permute(1,0,2).float().contiguous()
        v=record['v'][0].index_select(0,index).permute(1,0,2).float().contiguous()
        score=torch.bmm(q,k.transpose(1,2))*record['softmax_scale']
        zs.append(torch.logsumexp(score,-1));outputs.append(torch.bmm(score.softmax(-1),v))
    if sorted(keys)!=list(range(record['k'].shape[1])):raise ValueError('partition is not complete and disjoint')
    z=torch.stack(zs,-1);values=torch.stack(outputs,-2);mass=z.softmax(-1)
    full=(mass[...,None]*values).sum(-2);native=record['native_output'][0].permute(1,0,2).float()
    error=output_error(full,native);rows={}
    for i,name in enumerate(names):
        other=[j for j in range(len(names)) if j!=i]
        without=(z[...,other].softmax(-1)[...,None]*values[...,other,:]).sum(-2)
        density=len(parts[name])/record['k'].shape[1]
        sensitivity=(without-full).norm(dim=-1)/full.norm(dim=-1).clamp_min(1e-12)
        rows[name]=dict(tokens=len(parts[name]),key_density=density,mass=summary(mass[...,i]),
            mass_over_uniform_density=summary(mass[...,i]/density),remove_output_sensitivity=summary(sensitivity),
            lower_center_mass=summary(mass[:,3::4,i]),lower_center_remove_sensitivity=summary(sensitivity[:,3::4]))
    return dict(layer=record['layer'],phase=record['phase'],FP32_error=error,
        FP32_gate=error['max_abs']<=.02 and error['relative_l2']<=.01 and error['one_minus_cosine']<=.001,parts=rows)


def main():
    import av
    from PIL import Image,ImageDraw
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--source-video',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);source=json.loads((args.capture/'summary.json').read_text())
    if source['status']!='pass' or not source.get('observer_noise_latent_RGB_equivalence'):raise ValueError('unverified capture')
    indices=[4*f for f in range(40,48)];images={}
    with av.open(str(args.source_video)) as container:
        for i,frame in enumerate(container.decode(video=0)):
            if i in indices:images[i]=frame.to_ndarray(format='rgb24')
            if i==188:break
    if set(images)!=set(indices):raise ValueError('missing completed source pixels')
    masks=[];all_red=[];background=[];rng=np.random.default_rng(20260909)
    for i in indices:
        fraction,red,pile=red_cells(images[i]);nonred=np.flatnonzero(~red.flatten());n=int(pile.sum())
        if not n or len(nonred)<n:raise ValueError('red proxy/equal-token control not feasible')
        bg=np.zeros(880,dtype=bool);bg[rng.choice(nonred,n,replace=False)]=True
        masks.append(pile.flatten());all_red.append(red.flatten());background.append(bg)
    pile=np.concatenate(masks);red=np.concatenate(all_red);bg=np.concatenate(background)
    partitions=dict(initial=list(range(7040)),source_red_pile=(np.flatnonzero(pile)+7040).tolist(),
        source_other_red=(np.flatnonzero(red&~pile)+7040).tolist(),
        matched_nonred=(np.flatnonzero(bg)+7040).tolist(),
        remaining_source_nonred=(np.flatnonzero(~red&~bg)+7040).tolist(),current=list(range(14080,21120)))
    path=args.capture/'attention_teacher.pt'
    with path.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
    if digest!=source['attention_teacher']['sha256']:raise ValueError('capture changed')
    records=torch.load(path,map_location='cpu',weights_only=True)['records'];rows=[]
    for record in records:
        if record['k'].shape[1]!=21120:raise ValueError('this proxy audit is for reset_reveal only')
        rows.append(partition_record(record,partitions))
    overlay=Image.fromarray(images[188]).convert('RGB');draw=ImageDraw.Draw(overlay)
    for mask,color in ((masks[-1],'red'),(background[-1],'cyan')):
        for t in np.flatnonzero(mask):
            y,x=divmod(int(t),40);draw.rectangle((x*32,y*32,(x+1)*32-1,(y+1)*32-1),outline=color,width=2)
    overlay.save(args.output/'past_source_proxy.png')
    report=dict(schema='past_color_proxy_attention_diagnostic_v1',source_pixel_indices=indices,capture_sha256=digest,
        source_video=str(args.source_video),proxy='HSV red threshold and largest connected token-cell component',
        limitations=['not a semantic segmentation model','token receptive fields extend beyond pixel cells',
                     'local output sensitivity is not video quality','one prompt/seed and sampled queries only'],
        selected_pile_tokens=int(pile.sum()),matched_background_tokens=int(bg.sum()),source_total_tokens=7040,
        original_frame64_block_budget_not_used=True,partitions=partitions,rows=rows)
    (args.output/'region_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(pile_tokens=report['selected_pile_tokens'],total_source_tokens=7040,records=len(rows),
        all_FP32_gates=all(r['FP32_gate'] for r in rows))))


if __name__=='__main__':main()
