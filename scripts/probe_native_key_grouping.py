#!/usr/bin/env python3
"""B-line controlled key organization; immutable source partitions across captured phases."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_balanced_key_groups import balanced_key_groups
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
    partitions={};builds=[];source_hashes={};rows=[]
    protected=torch.tensor(list(range(7040))+list(range(14080,28160)),device='cuda')
    for ri,r in enumerate(data['records']):
        if r['query_frame']!=96 or list(r['q'].shape)!=[1,32,24,128] or r['k'].shape[1]!=28160:raise ValueError('unqualified capture')
        digest=hashlib.sha256(r['k'][:,7040:14080].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        if r['layer'] in source_hashes and source_hashes[r['layer']]!=digest:raise ValueError('source bank changed between phases')
        source_hashes[r['layer']]=digest
        q,k,v=[r[n].cuda() for n in ('q','k','v')]
        if r['layer'] not in partitions:
            if r['phase']!=0:raise ValueError('partition must be created from the first available source snapshot')
            bank=k[0,7040:14080]
            current={'spatial8':groups_for_source(22,40,kind='spatial8'),
                     'flat55':[list(range(i,i+55)) for i in range(0,7040,55)]}
            for kind in ('key_frame','key_bank'):
                torch.cuda.synchronize();began=time.perf_counter()
                if kind=='key_frame':
                    current[kind]=[[i+f*880 for i in g] for f in range(8) for g in balanced_key_groups(bank[f*880:(f+1)*880])]
                else:current[kind]=balanced_key_groups(bank)
                torch.cuda.synchronize();elapsed=time.perf_counter()-began
                assert sorted(map(len,current[kind]))==[55]*128
                builds.append(dict(layer=r['layer'],kind=kind,build_s=elapsed,groups=128,tokens_per_group=55,
                    includes_CPU_control_GPU_work_and_index_D2H=True))
            partitions[r['layer']]=current
        qh=q[0].permute(1,0,2).float().contiguous();kh=k[0].permute(1,0,2).float().contiguous();vh=v[0].permute(1,0,2).float().contiguous()
        full=(qh@kh.transpose(-1,-2)*(128**-.5)).softmax(-1)@vh
        error=output_error(full,r['native_output'][0].permute(1,0,2).cuda())
        gate=error['max_abs']<=.02 and error['relative_l2']<=.01 and error['one_minus_cosine']<=.001
        for kind,groups in partitions[r['layer']].items():
            ids=[torch.tensor([7040+i for i in g],device='cuda') for g in groups]
            counts=torch.tensor([len(g) for g in groups],device='cuda')
            km,vm=[torch.stack([x[0].index_select(0,index).float().mean(0) for index in ids]) for x in (k,v)]
            scores=source_head_scores(q[0],km,vm,counts,'mass_value').cpu().tolist()
            selected=torch.tensor([exact_source_indices(groups,s,1760) for s in scores],device='cuda')
            keep=torch.cat([protected[None].expand(24,-1),selected+7040],dim=1).sort(dim=1).values
            heads=torch.arange(24,device='cuda')[:,None];rk=kh[heads,keep];rv=vh[heads,keep]
            actual=(qh@rk.transpose(-1,-2)*(128**-.5)).softmax(-1)@rv
            residual={}
            for label,raw,mean in [('K',k[0,7040:14080],km),('V',v[0,7040:14080],vm)]:
                energy=raw.float().square().sum(-1).mean()
                represented=(mean.square().sum(-1).mean(-1)*counts).sum()/counts.sum()
                residual[label+'_within_group_variance_fraction']=float((energy-represented)/energy)
            row=dict(capture_row=ri,layer=r['layer'],phase=r['phase'],grouping=kind,groups=len(groups),
                selected_source_tokens_per_head=1760,policy='mass_value',head_policy='per_head',
                actual_deleted_output_error=output_error(full,actual),native_reference_gate=gate,native_reference_error=error,**residual)
            rows.append(row);print(json.dumps(row),flush=True)
        del q,k,v,qh,kh,vh,full,km,vm,rk,rv,actual
    with args.capture.open('rb') as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
    report=dict(schema='native_key_grouping_probe_v1',status='offline_diagnostic_complete',rows=rows,builds=builds,
        capture_sha256=digest,partitions_created_once_per_layer_and_reused_across_phases=True,
        controls='flat55/key_frame/key_bank all have128 groups of55 tokens; spatial8 remains a separate reference',
        limits=['past K only; no Q/teacher used for group construction',
            'balanced cosine bisection is a fixed-cap candidate, not the complete ClusterVLM algorithm',
            'one existing bead trajectory,32 geometric queries,3 layers; no video or full-Q claim',
            'partition created after recorded rebinding; live archive-time equivalence remains to be tested',
            'construction must be paid; group count/variance improvement alone is not generation quality'])
    (args.output/'key_grouping.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
