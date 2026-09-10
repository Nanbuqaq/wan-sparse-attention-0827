#!/usr/bin/env python3
"""Offline2x2: source-only/joint context normalization x mass/contrast proxy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_causal_block_memory import groups_for_source,source_head_scores,exact_source_indices
from adapters.longlive_sparse.native_source_context_proxy import source_context_scores
from scripts.analyze_native_attention_teacher import output_error


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=False)
    data=torch.load(args.capture,weights_only=True,map_location='cpu',mmap=True)
    if not data.get('online_routing_may_not_access'):raise ValueError('offline boundary flag required')
    source_groups=groups_for_source(22,40,kind='spatial8')
    protected=list(range(7040))+list(range(14080,28160))
    source_ids=[torch.tensor([7040+i for i in g],device='cuda') for g in source_groups]
    kept_groups=groups_for_source(22,40,frames=24,kind='flat64')
    kept_ids=[torch.tensor([protected[i] for i in g],device='cuda') for g in kept_groups]
    source_counts=torch.tensor([len(g) for g in source_groups],device='cuda')
    kept_counts=torch.tensor([len(g) for g in kept_groups],device='cuda')
    exact=torch.tensor(protected,device='cuda');rows=[]
    for ri,r in enumerate(data['records']):
        if r['query_frame']!=96 or list(r['q'].shape)!=[1,32,24,128] or r['k'].shape[1]!=28160:
            raise ValueError('unexpected native source capture geometry')
        q,k,v=[r[name].cuda() for name in ('q','k','v')]
        qh=q[0].permute(1,0,2).float().contiguous();kh=k[0].permute(1,0,2).float().contiguous();vh=v[0].permute(1,0,2).float().contiguous()
        full=(qh@kh.transpose(-1,-2)*(128**-.5)).softmax(-1)@vh
        err=output_error(full,r['native_output'][0].permute(1,0,2).cuda())
        gate=err['max_abs']<=.02 and err['relative_l2']<=.01 and err['one_minus_cosine']<=.001
        sk,sv=[torch.stack([x[0].index_select(0,ids).float().mean(0) for ids in source_ids]) for x in (k,v)]
        kk,kv=[torch.stack([x[0].index_select(0,ids).float().mean(0) for ids in kept_ids]) for x in (k,v)]
        for normalization in ('source_only','joint_context'):
            for policy in ('mass_value','contrast_value'):
                scores=(source_head_scores(q[0],sk,sv,source_counts,policy) if normalization=='source_only' else
                    source_context_scores(q[0],sk,sv,source_counts,kk,kv,kept_counts,policy)).cpu().tolist()
                chosen=torch.tensor([exact_source_indices(source_groups,s,1760) for s in scores],device='cuda')
                ids=torch.cat([exact[None].expand(24,-1),chosen+7040],dim=1).sort(dim=1).values
                heads=torch.arange(24,device='cuda')[:,None];rk=kh[heads,ids];rv=vh[heads,ids]
                actual=(qh@rk.transpose(-1,-2)*(128**-.5)).softmax(-1)@rv
                row=dict(capture_row=ri,layer=r['layer'],phase=r['phase'],normalization=normalization,policy=policy,
                    selected_source_tokens_per_head=1760,source_tokens_per_head=7040,protected_tokens_per_head=21120,
                    actual_deleted_output_error=output_error(full,actual),native_reference_gate=gate,native_reference_error=err,
                    selected_indices_sha256=hashlib.sha256(chosen.cpu().int().numpy().tobytes()).hexdigest())
                rows.append(row);print(json.dumps(row),flush=True)
        del q,k,v,qh,kh,vh,full,sk,sv,kk,kv,rk,rv,actual
    with args.capture.open('rb') as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
    out=dict(schema='native_source_context_probe_v1',status='offline_diagnostic_complete',rows=rows,capture_sha256=digest,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        candidate_inputs='current Q and small source/kept K/V means only; teacher is an independent actual deletion computation',
        scope='one bead trajectory,9 captures, geometric32 Q, per-head quarter source; not a video result',
        normalization_hypothesis='source-only scoring weights Q even when kept context dominates them; contrast reference also omits kept values',
        implementation_limits=['protected summaries have extra preparation cost not timed by this diagnostic',
            'source means after captured rebinding are not claimed bitwise equal to archived-mean rebinding',
            'no change to old moment/wire compression; this is deletion of native recalled raw source',
            'joint softmax/prototypes alone are not claimed novel'])
    (args.output/'source_context.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
