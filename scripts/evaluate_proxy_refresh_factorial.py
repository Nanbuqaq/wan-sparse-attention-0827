#!/usr/bin/env python3
"""Offline 2x2: raw/RoPE-aligned prototypes x first/current Q summaries."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.phase_prototypes import canonical_wan_frequency_table, archive_rope0_key
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer
from adapters.longlive_sparse.offline_eval import dense_history_attention, routed_history_attention, output_error_metrics
from scripts.audit_candidate_permutation import reconstruct
from scripts.evaluate_complete_attention_capture import retained_probability_mass


def construct_factorial(first, current, *, device):
    for name in ('key_unrotated','key','value','frame_ids','token_ids','history_positions'):
        if not torch.equal(first[name],current[name]):
            raise ValueError(f'factorial requires immutable identical history: {name}')
    if first['layer']!=current['layer'] or first['current_start']!=current['current_start']:
        raise ValueError('factorial must stay within one layer/chunk/trajectory')
    if first['denoising_pass']!=0 or current['denoising_pass']!=4:
        raise ValueError('this frozen factorial compares first and clean-context calls')
    if current['rope_policy']!='upstream_zero' or bool(current['history_positions'][...,0].any()):
        raise ValueError('index-time aligned prototype requires proved upstream_zero policy')
    archive,raw_current,frames=reconstruct(current)
    _,raw_first,first_frames=reconstruct(first)
    if frames!=first_frames:
        raise ValueError('candidate order must remain identical across refresh conditions')
    exact=current['exact_key'].shape[1]
    routes={'raw_first':archive.route_indexed(0,raw_first,frames,exact_k_tokens=exact),
            'raw_current':archive.route_indexed(0,raw_current,frames,exact_k_tokens=exact)}
    if routes['raw_first'].digest()!=first['route_sha']:
        raise RuntimeError('factorial raw-first control does not reproduce executed route')
    aligned=HistoryArchive(archive.config,spatial_height=archive.spatial_height,spatial_width=archive.spatial_width)
    freqs=canonical_wan_frequency_table(current['query'].shape[-1]).to(device)
    for frame in frames:
        index=archive._layers[0][frame]
        rotated=archive_rope0_key(index.key.to(device),spatial_height=archive.spatial_height,
                                  spatial_width=archive.spatial_width,freqs=freqs)
        mask=current['frame_ids'][0,0]==frame
        order=current['token_ids'][0,0,mask].argsort()
        if not torch.equal(rotated.cpu(),current['key'][:,mask][:,order]):
            raise RuntimeError('index-time raw-K rotation does not reproduce executed historical K')
        aligned.index_frame(0,frame,rotated,index.value,storage_k=index.key,storage_v=index.value)
        if not torch.equal(aligned._layers[0][frame].block_value_centroids,index.block_value_centroids):
            raise RuntimeError('alignment changed the V prototype control')
    for name,capture in (('aligned_first',first),('aligned_current',current)):
        summary=summarize_query_for_pretransfer(capture['query'].to(device),64)
        routes[name]=replace(aligned.route_indexed(0,summary,frames,exact_k_tokens=exact),method='rope0_final_capture_candidate')
    counts=[(p.union_frame_ids>=0).sum(-1) for p in routes.values()]
    if not all(torch.equal(counts[0],x) for x in counts[1:]):
        raise RuntimeError('factorial actual head-token budgets differ')
    return routes


def evaluate(first,current,*,device):
    routes=construct_factorial(first,current,device=device)
    q,k,v,ek,ev=[current[name].to(device) for name in ('query','key','value','exact_key','exact_value')]
    teacher=dense_history_attention(q,torch.cat((ek,k),1),torch.cat((ev,v),1))
    mass=retained_probability_mass(current,routes,device=device)
    records={}
    for name,plan in routes.items():
        output=routed_history_attention(q,k,v,current['frame_ids'],current['token_ids'],plan,exact_key=ek,exact_value=ev)
        records[name]={'output_error':output_error_metrics(teacher,output),
                      'probability_mass':mass[name],'selected_tokens':plan.unique_history_tokens,
                      'route_sha':plan.digest()}
    return {'status':'pass','scope':'offline_same_history_2x2_on_clean_context',
            'layer':current['layer'],'current_start':current['current_start'],'records':records,
            'history_immutable_verified':True,'actual_budget_matched':True,
            'aligned_K_reconstructed_from_raw_archive':True,'video_promotion_allowed':False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture-dir',required=True)
    parser.add_argument('--layer',required=True,type=int)
    parser.add_argument('--start',required=True,type=int)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path(args.capture_dir)
    paths=[root/f'layer{args.layer:02d}_start{args.start:08d}_pass{call:02d}.pt' for call in (0,4)]
    output=Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    captures=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
    result=evaluate(*captures,device=args.device)
    result['capture_sha256']=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:row['output_error'] for name,row in result['records'].items()},indent=2))


if __name__=='__main__':
    main()
