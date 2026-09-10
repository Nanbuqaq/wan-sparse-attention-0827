#!/usr/bin/env python3
"""Frozen query/time grouping candidates versus independently recomputed deletion outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_causal_block_memory import groups_for_source,source_head_scores,exact_source_indices
from scripts.analyze_native_attention_teacher import output_error


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(args.capture,map_location='cpu',weights_only=True,mmap=True)
    if not data.get('online_routing_may_not_access'):raise ValueError('offline teacher boundary absent')
    variants=[('spatial_mean','spatial8','mean'),('spatial_query_peak','spatial8','normalized_peak'),
        ('spacetime_mean','spacetime2x4','mean'),('flat_tube_matched_mean','flat_tube_matched','mean')]
    protected=torch.tensor(list(range(7040))+list(range(14080,28160)),device='cuda')
    heads=torch.arange(24,device='cuda')[:,None];rows=[]
    for ri,record in enumerate(data['records']):
        if (record['query_frame']!=96 or list(record['q'].shape)!=[1,32,24,128]
            or record['full_Q_tokens']!=7040 or record['k'].shape[1]!=28160):
            raise ValueError('unexpected source capture geometry')
        q,k,v=[record[name].cuda() for name in ('q','k','v')]
        qh=q[0].permute(1,0,2).float().contiguous()
        kh=k[0].permute(1,0,2).float().contiguous();vh=v[0].permute(1,0,2).float().contiguous()
        full=(qh@kh.transpose(-1,-2)*(128**-.5)).softmax(-1)@vh
        error=output_error(full,record['native_output'][0].permute(1,0,2).cuda())
        gate=error['max_abs']<=.02 and error['relative_l2']<=.01 and error['one_minus_cosine']<=.001
        for name,kind,reduction in variants:
            groups=groups_for_source(22,40,kind=kind)
            ids=[torch.tensor(g,device='cuda')+7040 for g in groups]
            counts=torch.tensor([len(g) for g in groups],device='cuda')
            km,vm=[torch.stack([raw[0].index_select(0,index).float().mean(0) for index in ids]) for raw in (k,v)]
            scores=source_head_scores(q[0],km,vm,counts,'mass_value',32,reduction).cpu().tolist()
            selected=torch.tensor([exact_source_indices(groups,s,1760) for s in scores],device='cuda')
            kept=torch.cat([protected[None].expand(24,-1),selected+7040],dim=1).sort(1).values
            # Independent teacher recomputes original-token attention after deletion,
            # including the changed softmax denominator. No proxy scores enter this calculation.
            kk,vv=kh[heads,kept],vh[heads,kept]
            actual=(qh@kk.transpose(-1,-2)*(128**-.5)).softmax(-1)@vv
            rows.append(dict(capture_row=ri,layer=record['layer'],phase=record['phase'],method=name,
                grouping=kind,query_reduction=reduction,groups=len(groups),source_tokens_per_head=1760,
                actual_deleted_output_error=output_error(full,actual),native_reference_gate=gate,
                native_reference_error=error,source_raw_KV_H2D_payload_bytes=1760*24*128*2*2,
                source_group_KV_mean_FP32_bytes=km.numel()*4*2,
                selected_indices_sha256=hashlib.sha256(selected.cpu().numpy().tobytes()).hexdigest()))
            print(json.dumps({k:rows[-1][k] for k in ('capture_row','method','actual_deleted_output_error','native_reference_gate')}),flush=True)
            del km,vm,kk,vv,actual
        del q,k,v,qh,kh,vh,full
    with args.capture.open('rb') as f:capture_sha=hashlib.file_digest(f,'sha256').hexdigest()
    result=dict(status='offline_diagnostic_complete',rows=rows,capture_sha256=capture_sha,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),GPU=torch.cuda.get_device_name(),
        online_inputs='current sampled Q and past source group K/V means/counts only',
        teacher='full original K/V fresh softmax after actual deletion; independent of proxy',
        limits=['one old bead trajectory; sampled geometric Q differs from live uniform32',
            'means formed after temporal rebind approximate live rebind-after-mean',
            'no full-Q or generation quality inference; failed native rows remain failed',
            'H2D bytes are logical raw payload not a measured transfer trace'])
    (args.output/'source_coverage.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
