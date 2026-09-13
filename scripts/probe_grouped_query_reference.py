#!/usr/bin/env python3
"""Same raw budget and value score: calibrated membership vs Block64/16 on real QKV."""
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.longlive_sparse.query_balanced_value import normalized_values,stratified_sites


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--membership',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    data=torch.load(a.case/'steady_observer.pt',weights_only=True,map_location='cpu',mmap=True);d=next(x for x in data['calls'] if x['frame']==88)
    membership=torch.load(a.membership,weights_only=True,map_location='cpu')['atom_group_ids'].numpy();ft=d['frame_tokens'];apf=ft//16
    clusters={};atoms=[]
    for pos,owner in d['eligible']:
        if owner[0]!='native' or owner[1]>=88:raise ValueError('not committed eligible history')
        for start in range(0,ft,16):
            group=int(membership[owner[1]*apf+start//16]);ids=list(range(pos*ft+start,pos*ft+start+16))
            clusters.setdefault(group,[]).extend(ids);atoms.append(ids)
    candidates={'Block64':d['token_blocks'],'calibrated_groups':sorted(clusters.values(),key=min),'Block16_granularity_control':atoms}
    keys=d['k'][0].float();values=d['v'][0].float();sites=stratified_sites(8,*d['token_grid']);q=d['q'][0,sites].float()
    budget=sum(map(len,d['token_blocks']))//2;full=[]
    for h in range(q.shape[1]):full.append((q[:,h]@keys[:,h].T/math.sqrt(q.shape[-1])).softmax(-1)@values[:,h])
    full=torch.stack(full);rows=[]
    for name,groups in candidates.items():
        began=time.perf_counter();km=torch.stack([keys[g].mean(0) for g in groups]);vm=torch.stack([values[g].mean(0) for g in groups]);counts=torch.tensor([len(g) for g in groups]);prototype_s=time.perf_counter()-began
        began=time.perf_counter();utility=normalized_values(q,km,vm,counts).sum(1).numpy();score_s=time.perf_counter()-began
        began=time.perf_counter();indices=[]
        for head in utility:
            selected=[];left=budget
            for group in np.argsort(-head,kind='stable'):
                count=min(left,len(groups[group]));selected.extend(sorted(groups[group])[:count]);left-=count
                if not left:break
            if left:raise RuntimeError('exact raw budget not filled')
            indices.append(sorted(d['protected']+selected))
        rank_index_s=time.perf_counter()-began
        outputs=[];pack_s=0.;attention_s=0.
        for h,ids in enumerate(indices):
            began=time.perf_counter();kk=keys[ids,h];vv=values[ids,h];pack_s+=time.perf_counter()-began
            began=time.perf_counter();outputs.append((q[:,h]@kk.T/math.sqrt(q.shape[-1])).softmax(-1)@vv);attention_s+=time.perf_counter()-began
        out=torch.stack(outputs);rows.append(dict(method=name,groups=len(groups),raw_budget_per_head=budget,
            CPU_prototype_build_s=prototype_s,CPU_value_score_s=score_s,CPU_rank_and_index_s=rank_index_s,CPU_raw_KV_pack_s=pack_s,CPU_FP32_attention_s=attention_s,
            relative_output_error=float((out-full).norm()/full.norm()),physical_union=len(set(x for ids in indices for x in ids)),
            original_KV_pack_write_bytes=2*len(indices)*len(indices[0])*q.shape[-1]*4))
    report=dict(status='pass',frame=88,queries_per_head=len(sites),heads=q.shape[1],rows=rows,
        scope='CPU fixed-input query/layout reference, not production GPU latency or video quality',
        common_selection='same normalized value sum score and exact .5 original-token budget; deterministic final-group prefix trim in every arm',
        not_identical_to_online_whole_group_policy=True,future_members_not_used='stable atom IDs clipped to actual eligible past tokens; prototypes recomputed from current captured rawKV only',
        feature_assignment_and_D2H_costs='separately recorded calibrated_groups_bird_v1, not hidden in query-only comparison',
        original_KV_executed_not_prototypes=True)
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
