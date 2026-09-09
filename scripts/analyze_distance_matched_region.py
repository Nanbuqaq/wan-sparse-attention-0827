#!/usr/bin/env python3
"""Offline red-region sensitivity with frame/distance-matched nonred controls.

Selection uses only source color-proxy labels and geometry, never attention
scores. This is not a causal online router or a video-quality experiment.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_native_attention_teacher import output_error


def distance_key(token,query_position,grid,bucket_squared):
    height,width=grid;frame_tokens=height*width
    frame,position=divmod(int(token),frame_tokens)
    y,x=divmod(position,width);qy,qx=divmod(int(query_position)%frame_tokens,width)
    return frame,((y-qy)**2+(x-qx)**2)//bucket_squared


def matched_indices(region,eligible,query_position,*,grid=(22,40),frames=8,bucket_squared=16,seed=20260910):
    region=np.asarray(region,dtype=np.int64);eligible=np.asarray(eligible,dtype=np.int64)
    total=frames*grid[0]*grid[1]
    if bucket_squared<1 or min(grid)<1 or frames<1:raise ValueError('positive geometry/bucket required')
    if any(len(np.unique(x))!=len(x) or np.any(x<0) or np.any(x>=total) for x in (region,eligible)):
        raise ValueError('token sets must be unique and within the source')
    if np.intersect1d(region,eligible).size:raise ValueError('region and control must be disjoint')
    foreground=defaultdict(list);background=defaultdict(list)
    for token in sorted(region):foreground[distance_key(token,query_position,grid,bucket_squared)].append(int(token))
    for token in sorted(eligible):background[distance_key(token,query_position,grid,bucket_squared)].append(int(token))
    rng=np.random.default_rng(seed);picked=[];control=[];strata=[]
    for key,values in sorted(foreground.items()):
        candidates=background.get(key,[]);count=min(len(values),len(candidates))
        # Same deterministic matchable foreground subset across control draws.
        picked.extend(values[:count])
        if count:control.extend(rng.choice(candidates,size=count,replace=False).tolist())
        strata.append(dict(frame=key[0],distance_bin=key[1],region_available=len(values),control_available=len(candidates),matched=count))
    return dict(region=sorted(picked),control=sorted(control),matched=len(picked),
                original_region=len(region),coverage=len(picked)/max(len(region),1),strata=strata)


def masked_attention(logits,values,removed):
    if removed.dtype!=torch.bool or tuple(removed.shape)!=tuple(logits.shape[1:]):
        raise ValueError('per-query removal mask shape/dtype mismatch')
    if removed.all(-1).any():raise ValueError('cannot remove every key')
    return torch.bmm(logits.masked_fill(removed.unsqueeze(0),-torch.inf).softmax(-1),values)


def summarize_error(values,valid):
    selected=values[:,valid].double().flatten()
    if not selected.numel():return dict(mean=None,p50=None,p95=None)
    return dict(mean=float(selected.mean()),p50=float(selected.median()),p95=float(torch.quantile(selected,.95)))


@torch.inference_mode()
def analyze_record(record,region,eligible,*,control_seeds,bucket_squared):
    if record.get('attention_causal_mask') or record.get('token_grid')!=[22,40] or record['frame_tokens']!=880:
        raise ValueError('qualified native source/query geometry changed')
    q=record['q'][0].permute(1,0,2).float().contiguous()
    k=record['k'][0].permute(1,0,2).float().contiguous()
    v=record['v'][0].permute(1,0,2).float().contiguous()
    if k.shape[1]!=21120 or k.shape!=v.shape:raise ValueError('this study requires initial/source/current partition')
    positions=record['query_indices'].cpu().tolist()
    logits=torch.bmm(q,k.transpose(1,2))*float(record['softmax_scale'])
    full=torch.bmm(logits.softmax(-1),v)
    native=record['native_output'][0].permute(1,0,2).float()
    numerical=output_error(full,native)
    gate=numerical['max_abs']<=.02 and numerical['relative_l2']<=.01 and numerical['one_minus_cosine']<=.001
    if not gate:raise ValueError(f'original FP32 replay gate failed: {numerical}')
    masks=[torch.zeros(q.shape[1],k.shape[1],dtype=torch.bool) for _ in range(1+len(control_seeds))]
    coverage=[];valid=[];by_site={}
    for qi,position in enumerate(positions):
        spatial=int(position)%880
        matches=[matched_indices(region,eligible,spatial,seed=seed+65537*spatial,bucket_squared=bucket_squared) for seed in control_seeds]
        assert all(x['region']==matches[0]['region'] for x in matches)
        coverage.append(matches[0]['coverage']);valid.append(matches[0]['matched']>0)
        masks[0][qi,torch.tensor(matches[0]['region'],dtype=torch.long)+7040]=True
        for index,match in enumerate(matches,1):masks[index][qi,torch.tensor(match['control'],dtype=torch.long)+7040]=True
        by_site[str(spatial)]=dict(matched=matches[0]['matched'],coverage=matches[0]['coverage'],strata=matches[0]['strata'])
    valid=torch.tensor(valid,dtype=torch.bool)
    norm=full.norm(dim=-1).clamp_min(1e-12)
    sensitivity=[(masked_attention(logits,v,mask)-full).norm(dim=-1)/norm for mask in masks]
    roi=summarize_error(sensitivity[0],valid);controls=[]
    for seed,values in zip(control_seeds,sensitivity[1:]):
        stats=summarize_error(values,valid)
        ratio=roi['mean']/stats['mean'] if stats['mean'] is not None and stats['mean']>0 else None
        controls.append(dict(seed=seed,sensitivity=stats,roi_over_control_mean_ratio=ratio))
    sites={}
    for i,name in enumerate(record['query_sites']):
        subset=valid & (torch.arange(q.shape[1])%len(record['query_sites'])==i)
        sites[name]=dict(roi=summarize_error(sensitivity[0],subset),
                        controls=[summarize_error(s,subset) for s in sensitivity[1:]])
    return dict(layer=record['layer'],phase=record['phase'],query_frame=record['query_frame'],
                FP32_gate=True,FP32_error=numerical,roi_sensitivity=roi,controls=controls,
                query_sites=sites,query_coverage=coverage,geometry_matches=by_site,
                valid_queries=int(valid.sum()),query_count=len(valid))


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--region-diagnostic',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    result=dict(status='running',scope='offline_color_proxy_distance_matched_sensitivity_not_video_quality',rows=[],
                torch=torch.__version__,CPU_threads=torch.get_num_threads(),
                analysis_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        source=json.loads((args.capture/'summary.json').read_text());old=json.loads(args.region_diagnostic.read_text())
        if source['status']!='pass' or not source.get('observer_noise_latent_RGB_equivalence'):
            raise ValueError('capture lacks complete trajectory equivalence')
        path=args.capture/'attention_teacher.pt'
        with path.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
        if digest!=old['capture_sha256'] or digest!=source['attention_teacher']['sha256']:
            raise ValueError('capture identity differs from the existing region study')
        parts=old['partitions'];region=np.asarray(parts['source_red_pile'])-7040
        eligible=np.asarray(parts['matched_nonred']+parts['remaining_source_nonred'])-7040
        assert old['source_total_tokens']==7040
        seeds=[20260910,20260911,20260912];bucket=16
        data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if data['schema']!='native_attention_teacher_v1' or not data['online_routing_may_not_access']:
            raise ValueError('not an isolated offline teacher')
        result.update(capture_sha256=digest,region_diagnostic_sha256=hashlib.sha256(args.region_diagnostic.read_bytes()).hexdigest(),
                      source_pixels=old['source_pixel_indices'],bucket_squared_tokens=bucket,control_seeds=seeds,
                      source_mask_is_past_color_proxy=True,unmatched_foreground_not_silently_retained=True,
                      limitations=['one source trajectory','fixed geometric query samples, not semantic query masks',
                                   'squared spatial distance is binned, not exact relative RoPE matching',
                                   'color-connected ROI may include falling beads and wide token receptive fields',
                                   'attention sensitivity is not video quality; no online score is frozen'])
        for record in data['records']:
            row=analyze_record(record,region,eligible,control_seeds=seeds,bucket_squared=bucket);result['rows'].append(row)
            print(json.dumps(dict(layer=row['layer'],phase=row['phase'],coverage=min(row['query_coverage']),
                                  ratios=[x['roi_over_control_mean_ratio'] for x in row['controls']])),flush=True)
        result['status']='pass'
    except Exception:
        result.update(status='fail',traceback=traceback.format_exc());raise
    finally:
        (args.output/'distance_matched_region.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
