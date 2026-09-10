#!/usr/bin/env python3
"""Offline query grouping/budget discrimination with independently normalized teacher."""
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


def query_partition(indices,kind,frame_tokens=880):
    values=indices.tolist()
    if kind=='shared':return [list(range(len(values)))]
    if kind=='frame_pairs':labels=[(x//frame_tokens)//2 for x in values]
    elif kind=='spatial_sites':
        sites=sorted({x%frame_tokens for x in values})
        labels=[sites.index(x%frame_tokens) for x in values]
    else:raise ValueError('unknown query partition')
    result=[[i for i,label in enumerate(labels) if label==g] for g in sorted(set(labels))]
    if len(result)!=4 or any(len(g)!=8 for g in result):
        raise ValueError('this capture protocol needs four groups of eight geometric Q samples')
    return result


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    args.output.mkdir(parents=True,exist_ok=False)
    data=torch.load(args.capture,map_location='cpu',weights_only=True,mmap=True)
    if not data.get('online_routing_may_not_access'):raise ValueError('offline boundary flag absent')
    groups=groups_for_source(22,40,kind='spatial8')
    counts=torch.tensor([len(g) for g in groups],device='cuda')
    group_ids=[torch.tensor([7040+i for i in group],device='cuda') for group in groups]
    protected=torch.tensor(list(range(7040))+list(range(14080,28160)),device='cuda')
    variants=[('shared',.125),('shared',.25),('shared',.5),('frame_pairs',.25),('spatial_sites',.25)]
    rows=[]
    for ri,record in enumerate(data['records']):
        if record['query_frame']!=96 or list(record['q'].shape)!=[1,32,24,128] or record['full_Q_tokens']!=7040:
            raise ValueError('unexpected capture geometry')
        q,k,v=[record[name].cuda() for name in ('q','k','v')]
        if k.shape[1]!=28160:raise ValueError('source must occupy native second8-frame slot')
        qh=q[0].permute(1,0,2).float().contiguous()
        kh=k[0].permute(1,0,2).float().contiguous();vh=v[0].permute(1,0,2).float().contiguous()
        # Teacher reads complete original tensors; candidate uses only Q plus group means.
        full=(qh@kh.transpose(-1,-2)*(128**-.5)).softmax(-1)@vh
        native=record['native_output'][0].permute(1,0,2).cuda()
        error=output_error(full,native)
        passed=error['max_abs']<=.02 and error['relative_l2']<=.01 and error['one_minus_cosine']<=.001
        km,vm=[torch.stack([tensor[0].index_select(0,ids).float().mean(0) for ids in group_ids]) for tensor in (k,v)]
        for grouping,fraction in variants:
            partitions=query_partition(record['query_indices'],grouping)
            output=torch.empty_like(full);selected_by_head=[set() for _ in range(24)]
            route_hash=hashlib.sha256();budget=int(7040*fraction)
            for sites in partitions:
                qids=torch.tensor(sites,device='cuda')
                scores=source_head_scores(q[0].index_select(0,qids),km,vm,counts,'mass_value',samples=len(sites)).cpu().tolist()
                chosen=[exact_source_indices(groups,score,budget) for score in scores]
                route_hash.update(torch.tensor(chosen,dtype=torch.int32).numpy().tobytes())
                for head,ids in enumerate(chosen):selected_by_head[head].update(ids)
                # Recompute deletion output from original KV with fresh softmax normalization.
                source_ids=torch.tensor(chosen,device='cuda')+7040
                ids=torch.cat([protected[None].expand(24,-1),source_ids],dim=1).sort(dim=1).values
                heads=torch.arange(24,device='cuda')[:,None]
                remaining_k=kh[heads,ids];remaining_v=vh[heads,ids]
                qq=qh.index_select(1,qids)
                actual=(qq@remaining_k.transpose(-1,-2)*(128**-.5)).softmax(-1)@remaining_v
                output.index_copy_(1,qids,actual)
            unique=sum(len(ids) for ids in selected_by_head)
            # Per-head union can be shared by Q groups; repeated copies and padding are separate.
            row=dict(capture_row=ri,layer=record['layer'],phase=record['phase'],query_grouping=grouping,
                source_fraction_per_query=fraction,source_tokens_per_query_head=budget,query_groups=len(partitions),
                original_source_tokens_per_head=7040,unique_source_head_token_pairs=unique,
                raw_union_transfer_fraction=unique/(24*7040),raw_union_KV_payload_bytes=unique*128*2*2,
                repeated_group_raw_KV_payload_bytes=len(partitions)*24*budget*128*2*2,
                max_head_union_padding_payload_bytes=max(map(len,selected_by_head))*24*128*2*2,
                executed_attention_density=(21120+budget)/28160,
                actual_deleted_output_error=output_error(full,output),native_reference_gate=passed,
                native_reference_error=error,selected_indices_sha256=route_hash.hexdigest())
            rows.append(row);print(json.dumps(row),flush=True)
        del q,k,v,qh,kh,vh,full,native,km,vm,output,remaining_k,remaining_v,actual
    with args.capture.open('rb') as h:capture_sha=hashlib.file_digest(h,'sha256').hexdigest()
    result=dict(schema='native_query_groups_probe_v1',status='offline_diagnostic_complete',rows=rows,
        capture_sha256=capture_sha,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        online_inputs='only sampled current Q and source group K/V means; complete KV/output reserved for independent teacher',
        scoring='mass_value, per-head, spatial8 source grouping',
        limits=['one existing bead trajectory, layers0/14/29, phases0/3/clean4',
                '32 geometric Q samples, not full-Q or the live uniform32 policy',
                'means from already rebound K approximate archive-mean rebinding; not asserted bitwise same live route',
                'raw union transfer is a payload lower bound, not measured total IO or actual sparse backend cost',
                'query-specific routing changes H2D union at fixed Attention budget',
                'no video quality or production implementation claim; failed numeric rows cannot select layer policy'])
    (args.output/'query_groups.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
