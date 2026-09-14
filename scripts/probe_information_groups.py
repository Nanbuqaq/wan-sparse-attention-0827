#!/usr/bin/env python3
"""Causal clean-stream membership replay with original raw-KV output checks."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.information_group_selection import InformationGroupSelector
from adapters.longlive_sparse.query_balanced_value import normalized_values,stratified_sites,execute_per_head


def reduce16(values):
    # One original880-token frame:13 full64 blocks plus one48 tail.
    if values.shape[0]!=55:raise ValueError('native5B Block16 frame differs')
    return torch.cat([values[:52].reshape(13,4,*values.shape[1:]).mean(1),values[52:].mean(0,keepdim=True)])


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    rows=[];streams=[]
    for case in args.cases:
        data=torch.load(case/'steady_observer.pt',map_location='cpu',weights_only=True,mmap=True)
        captures={x['frame']:x for x in data['calls']};arrivals=iter(data['arrivals']);arrival=next(arrivals,None)
        bank={};seen=set();selectors={k:InformationGroupSelector(k) for k in ('flat_exact','physical16','value16')}
        for access in data['accesses']:
            frame=access['frame']
            if 'eligible' not in access or frame in seen or frame>88:continue
            seen.add(frame)
            while arrival is not None and arrival['frame']+8<=frame:
                for j,owner in enumerate(arrival['owners']):
                    bank[tuple(owner)]=(reduce16(arrival['key_mean'][j*55:(j+1)*55]).cuda(),
                                        reduce16(arrival['value_mean'][j*55:(j+1)*55]).cuda())
                arrival=next(arrivals,None)
            owners=[tuple(o) for _,o in access['eligible']]
            if any(o[0]!='native' or o[1]>=frame for o in owners):raise RuntimeError('future/current owner entered group producer')
            if any(o not in bank for o in owners):raise RuntimeError('missing past committed prototype')
            km=torch.cat([bank[o][0] for o in owners]);vm=torch.cat([bank[o][1] for o in owners])
            costs=[c for _ in owners for c in ([64]*13+[48])];counts=torch.tensor(costs,device='cuda')
            keys=[(o,b) for o in owners for b in range(14)]
            torch.cuda.synchronize();began=time.perf_counter()
            selectors['value16'].value_groups.update(14,keys,vm,counts)
            torch.cuda.synchronize();update_s=time.perf_counter()-began
            streams.append(dict(case=case.name,frame=frame,eligible_frames=[o[1] for o in owners],
                                blocks=len(keys),group_update_wall_s=update_s,past_only=True))
            bank={k:v for k,v in bank.items() if k in set(owners)}
            if frame not in captures:continue
            d=captures[frame];q,k,v=[d[x].cuda() for x in ('q','k','v')]
            # Membership is a causal replay of recorded Block16 means. Score
            # uses the original captured Block64 means, not the conversion.
            ckm,cvm=[d[x].cuda() for x in ('key_mean','value_mean')]
            conversion_error=max(float((km-ckm).abs().max()),float((vm-cvm).abs().max()))
            if conversion_error>1e-4:raise RuntimeError('Block16-to64 conversion differs materially')
            mapping=torch.full((k.shape[1],),-2,dtype=torch.long);mapping[d['protected']]=-1
            for b,ids in enumerate(d['token_blocks']):mapping[ids]=b
            if (mapping==-2).any():raise RuntimeError('incomplete physical graph')
            mapping=mapping.cuda();sites=stratified_sites(8,*d['token_grid']).cuda()
            a=normalized_values(q[0,sites],ckm,cvm,counts);budget=sum(costs)//2
            qq=q[0,sites].float().permute(1,0,2)
            dense=(((qq@k[0].float().permute(1,2,0))/128**.5).softmax(-1)@v[0].float().permute(1,0,2)).permute(1,0,2)
            reference=None
            for kind,selector in selectors.items():
                # Prime fixed budget geometry, then measure one warm route+pack.
                selector.select(a,costs,budget,cvm,keys,14)
                torch.cuda.synchronize();began=time.perf_counter()
                mask,coverage,used,gids=selector.select(a,costs,budget,cvm,keys,14)
                torch.cuda.synchronize();select_s=time.perf_counter()-began
                if not torch.equal((mask*counts).sum(1),used):raise RuntimeError('exact pair budget violated')
                if reference is None:reference=mask.clone()
                began=time.perf_counter();out,visible=execute_per_head(q,k,v,mask,mapping,len(d['protected'])+budget)
                torch.cuda.synchronize();execute_s=time.perf_counter()-began
                numerator=denominator=maximum=0.
                for h in range(24):
                    kk=k[0,visible[h],h].float();vv=v[0,visible[h],h].float()
                    teacher=((q[0,sites,h].float()@kk.T)/128**.5).softmax(-1)@vv
                    error=out[0,sites,h].float()-teacher
                    numerator+=float(error.square().sum());denominator+=float(teacher.square().sum())
                    maximum=max(maximum,float(error.abs().max()))
                relative=(numerator/max(denominator,1e-30))**.5
                if relative>.01 or maximum>.02:raise RuntimeError('independent selected-output gate failed')
                cpu=mask.cpu();torch.save(cpu,args.output/f'{case.name}_{frame}_{kind}_route.pt')
                row=dict(case=case.name,frame=frame,kind=kind,actual_pairs=int(visible.sum())*q.shape[1],
                    changed_from_flat_exact=int((mask!=reference).sum()),changed_from_old_fast=int((cpu!=d['selected_group_mask']).sum()),
                    selected_tokens_per_head=used.cpu().tolist(),physical_union_tokens=int(visible.any(0).sum()),
                    warm_selection_wall_s=select_s,pack_FA2_wall_s=execute_s,
                    independent_selected_relative_L2=relative,independent_selected_max_abs=maximum,
                    independent_dense_relative_L2=float(((out[0,sites].float()-dense).square().sum()/dense.square().sum().clamp_min(1e-30)).sqrt()),
                    prototype_conversion_max_abs=conversion_error,
                    route_sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
                    selector=selector.audit())
                rows.append(row);print(json.dumps({x:row[x] for x in ('case','frame','kind','changed_from_flat_exact','warm_selection_wall_s','independent_dense_relative_L2')}),flush=True)
            del q,k,v,ckm,cvm,a,qq,dense,out,visible,reference
    result=dict(status='pass',GPU=torch.cuda.get_device_name(),rows=rows,stream_updates=streams,
        scope='real5B causal metadata and fixed-QKV replay, no video quality or production speed claim',
        no_current_Q_in_group_membership=True,no_MSE_promotion_gate=True,
        source_backend='original per-head FA2; original rawKV and positions',fallback=False,
        source_code_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('scripts/probe_information_groups.py','adapters/longlive_sparse/information_group_selection.py')})
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
