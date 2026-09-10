#!/usr/bin/env python3
"""Offline F/D discrimination: deletion vs explicit multiplicity, frozen vs fresh route."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_causal_block_memory import groups_for_source,source_head_scores,exact_source_indices
from scripts.analyze_native_attention_teacher import output_error


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(args.capture,weights_only=True,map_location='cpu',mmap=True)
    if not data.get('online_routing_may_not_access'):raise ValueError('offline boundary flag required')
    groups=groups_for_source(22,40,kind='spatial8');counts=torch.tensor([len(g) for g in groups],device='cuda')
    group_ids=[torch.tensor([7040+i for i in g],device='cuda') for g in groups]
    protected=torch.tensor(list(range(7040))+list(range(14080,28160)),device='cuda')
    frozen={};source_hashes={};unions={};rows=[]
    for ri,r in enumerate(data['records']):
        if r['query_frame']!=96 or list(r['q'].shape)!=[1,32,24,128] or r['k'].shape[1]!=28160:
            raise ValueError('unexpected capture geometry')
        key=(r['layer'],)
        source_hash=hashlib.sha256()
        for name in ('k','v'):source_hash.update(r[name][:,7040:14080].contiguous().view(torch.uint8).numpy().tobytes())
        digest=source_hash.hexdigest()
        if key in source_hashes and source_hashes[key]!=digest:raise ValueError('source KV changed across phases; cannot reuse same-bank coordinates')
        source_hashes[key]=digest
        q,k,v=[r[name].cuda() for name in ('q','k','v')]
        qh=q[0].permute(1,0,2).float().contiguous();kh=k[0].permute(1,0,2).float().contiguous();vh=v[0].permute(1,0,2).float().contiguous()
        full_prob=(qh@kh.transpose(-1,-2)*(128**-.5)).softmax(-1);full=full_prob@vh
        full_source_mass=full_prob[:,:,7040:14080].sum(-1)
        err=output_error(full,r['native_output'][0].permute(1,0,2).cuda())
        gate=err['max_abs']<=.02 and err['relative_l2']<=.01 and err['one_minus_cosine']<=.001
        km,vm=[torch.stack([x[0].index_select(0,ids).float().mean(0) for ids in group_ids]) for x in (k,v)]
        scores=source_head_scores(q[0],km,vm,counts,'mass_value').cpu().tolist()
        fresh=torch.tensor([exact_source_indices(groups,s,1760) for s in scores],device='cuda')
        if key not in frozen:
            if r['phase']!=0:raise ValueError('phase0 must establish frozen route before later phases')
            frozen[key]=fresh.clone();unions[key]=[set() for _ in range(24)]
        for head,ids in enumerate(fresh.cpu().tolist()):unions[key][head].update(ids)
        for route_name,chosen in [('frozen_phase0',frozen[key]),('fresh_for_capture',fresh)]:
            for multiplicity in (1,4):
                source_ids=(chosen+7040).repeat_interleave(multiplicity,dim=1)
                ids=torch.cat([protected[None].expand(24,-1),source_ids],dim=1).sort(dim=1).values
                heads=torch.arange(24,device='cuda')[:,None];rk=kh[heads,ids];rv=vh[heads,ids]
                probability=(qh@rk.transpose(-1,-2)*(128**-.5)).softmax(-1)
                actual=probability@rv
                source_mask=(ids>=7040)&(ids<14080)
                mass=(probability*source_mask[:,None,:]).sum(-1)
                row=dict(capture_row=ri,layer=r['layer'],phase=r['phase'],route=route_name,source_multiplicity=multiplicity,
                    raw_source_unique_tokens_per_head=1760,executed_source_instances_per_head=1760*multiplicity,
                    executed_total_K=ids.shape[1],executed_attention_density=ids.shape[1]/28160,
                    raw_recall_fraction=.25,full_source_mass_mean=float(full_source_mass.mean()),selected_source_mass_mean=float(mass.mean()),
                    source_mass_error_relative_l2=float((mass-full_source_mass).norm()/full_source_mass.norm().clamp_min(1e-20)),
                    actual_output_error=output_error(full,actual),native_reference_gate=gate,native_reference_error=err)
                rows.append(row);print(json.dumps(row),flush=True)
        del q,k,v,qh,kh,vh,full_prob,full,km,vm,rk,rv,probability,actual
    with args.capture.open('rb') as h:sha=hashlib.file_digest(h,'sha256').hexdigest()
    out=dict(schema='native_source_multiplicity_probe_v1',status='offline_diagnostic_complete',rows=rows,capture_sha256=sha,
        source_KV_unchanged_across_captured_phases=True,
        fresh_route_observed_three_phase_union_fraction={str(k[0]):sum(map(len,s))/(24*7040) for k,s in unions.items()},
        limits=['one bead trajectory and sampled32 geometric Q; not live uniform32',
            'phase1/2 are not captured; observed union is not complete four-step transfer cost',
            'multiplicity4 explicitly executes repeated original KV and full native K count; it is not quarter Attention work',
            'raw selected payload fixed25 percent; GPU replication and full Attention costs not timed here',
            'fixed inverse-fraction weighting is not exact source-mass matching for a deterministic top selection',
            'actual repeated-KV softmax recomputed independently; no utility self-validation or video claim'])
    (args.output/'multiplicity.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
