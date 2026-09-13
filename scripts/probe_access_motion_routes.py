#!/usr/bin/env python3
"""Real5B fixed-input route diagnostics; independent selected-output teacher.

No MSE-based promotion gate. Stage replay reuses the captured Q only to check
quota feasibility; it is explicitly not evidence across denoising stages.
"""
import argparse,json,sys,time,hashlib
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.longlive_sparse.query_balanced_value import normalized_values,stratified_sites,select_static_once,execute_per_head
from adapters.longlive_sparse.access_motion_selectors import select_recent_bridge,select_value_novelty,stage_budgets


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frame',type=int,choices=(24,88),required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    payload=torch.load(args.case/'steady_observer.pt',weights_only=True,map_location='cpu',mmap=True)
    d=next(x for x in payload['calls'] if x['frame']==args.frame)
    q,k,v,km,vm,count=[d[x].cuda() for x in ('q','k','v','key_mean','value_mean','counts')]
    costs=d['counts'].tolist();budget=sum(costs)//2;ft=d['frame_tokens']
    mapping=torch.full((k.shape[1],),-2,dtype=torch.long);mapping[d['protected']]=-1
    for g,indices in enumerate(d['token_blocks']):mapping[indices]=g
    if (mapping==-2).any():raise RuntimeError('incomplete graph')
    mapping=mapping.cuda();sites=stratified_sites(8,*d['token_grid']).cuda()
    aa=normalized_values(q[0,sites],km,vm,count)
    # Physical frame order is chronological for optional native rolling frames;
    # verify against actual owner coordinates instead of assuming global order.
    eligible={pos:owner for pos,owner in d['eligible']}
    group_frames=[eligible[indices[0]//ft][1] for indices in d['token_blocks']]
    reference=select_static_once(aa,costs,budget)[0]
    if not torch.equal(reference.cpu(),d['selected_group_mask']):raise RuntimeError('source route not reproduced')
    candidates={
        'sum_fast':lambda:select_static_once(aa,costs,budget),
        'recent_bridge':lambda:select_recent_bridge(aa,costs,budget,group_frames,ft),
        'value_novelty':lambda:select_value_novelty(aa,costs,budget,vm),
        'quota_quarter':lambda:select_static_once(aa,costs,sum(costs)//4),
        'quota_three_quarters':lambda:select_static_once(aa,costs,3*sum(costs)//4)}
    rows=[];usage={}
    for name,fn in candidates.items():
        torch.cuda.synchronize();started=time.perf_counter();mask,coverage,used=fn();torch.cuda.synchronize()
        select_s=time.perf_counter()-started;usage[name]=used.cpu()
        limit=int(used.max().cpu())+len(d['protected'])
        started=time.perf_counter();out,visible=execute_per_head(q,k,v,mask,mapping,limit);torch.cuda.synchronize()
        execute_s=time.perf_counter()-started
        if not torch.isfinite(out).all():raise RuntimeError('nonfinite rawKV output')
        numerator=denominator=0.
        # Independent actual selected Attention including re-normalization.
        for h in range(q.shape[2]):
            kk=k[0,visible[h],h].float();vv=v[0,visible[h],h].float()
            teacher=(q[0,sites,h].float()@kk.T/q.shape[-1]**.5).softmax(-1)@vv
            numerator+=float((out[0,sites,h].float()-teacher).square().sum())
            denominator+=float(teacher.square().sum())
        error=(numerator/max(denominator,1e-30))**.5
        if error>.02:raise RuntimeError('raw selected FA2 numerical gate failed')
        cpu=mask.cpu();torch.save(cpu,args.output/(name+'_route.pt'))
        rows.append(dict(method=name,used_tokens_per_head=used.cpu().tolist(),changed_groups=int((mask!=reference).sum()),
            select_wall_s=select_s,pack_FA2_wall_s=execute_s,FP32_selected_relative_L2=error,
            actual_pairs=int(visible.sum())*q.shape[1],physical_union_tokens=int(visible.any(0).sum()),
            route_sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest()))
    actual=2*(usage['quota_quarter']+usage['quota_three_quarters'])
    control=4*usage['sum_fast']
    report=dict(status='pass',source=str(args.case),frame=args.frame,GPU=torch.cuda.get_device_name(),
        q_shape=list(q.shape),k_shape=list(k.shape),rows=rows,
        quota_stage_totals_match_each_head=bool(torch.equal(actual,control)),
        uniform_four_call_tokens=control.tolist(),split_four_call_tokens=actual.tolist(),
        registered_quotas=stage_budgets(sum(costs),ft,'early_heavy'),
        stage_scope='single real first-denoise Q reused for quota feasibility, not real early/late dynamics',
        prototype_scope='past committed source observer prototypes; production CPU mirror/transfer costs not implemented for novelty',
        timing_scope='single synchronized diagnostic; no repeated speed or video quality claim',
        teacher_scope='independent FP32 actual selected output, never selector input',fallback=False)
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
