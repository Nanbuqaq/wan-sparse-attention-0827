#!/usr/bin/env python3
"""Query-balanced versus sum-value reference on one real, matched resident input."""
import argparse,json,math,time
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.query_balanced_value import stratified_sites,normalized_values,select_reference
from wave2_capture_io import load_call


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--legacy-source',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    d,report=load_call(args.case,legacy_source=args.legacy_source)
    height,width=[x//2 for x in report['latent_shape'][-2:]];sites=stratified_sites(8,height,width)
    q=d['q'][0,sites].float();k=d['k'][0].float();v=d['v'][0].float();a=normalized_values(q,d['key_mean'],d['value_mean'],d['counts'])
    costs=d['counts'].tolist();budget=math.floor(sum(costs)*.5);rows=[];chosen={};started=time.perf_counter()
    full=torch.stack([(q[:,h]@k[:,h].T/math.sqrt(q.shape[-1])).softmax(-1)@v[:,h] for h in range(q.shape[1])])
    for balanced in (False,True):
        name='query_balanced' if balanced else 'matched_sum';began=time.perf_counter();groups=[];cov=[];used=[]
        for head in a:
            ids,c,u=select_reference(head.numpy(),costs,budget,balanced=balanced);groups.append(ids);cov.extend(c.tolist());used.append(u)
        score_s=time.perf_counter()-began;outputs=[];union=set()
        for h,ids in enumerate(groups):
            selected=sorted(d['protected']+[t for b in ids for t in d['token_blocks'][b]]);union.update(selected)
            outputs.append((q[:,h]@k[selected,h].T/math.sqrt(q.shape[-1])).softmax(-1)@v[selected,h])
        output=torch.stack(outputs);chosen[name]=groups
        rows.append(dict(method=name,selection_CPU_s=score_s,source_tokens_per_head=used,
            physical_token_union=len(union),coverage_min=min(cov),coverage_p05=float(np.quantile(cov,.05)),coverage_mean=float(np.mean(cov)),
            clipped_objective=float(np.minimum(.8,cov).sum()),independent_FP32_output_relative_L2=float((output-full).norm()/full.norm())))
    result=dict(status='pass',input_scope=d.get('input_scope','new steady input'),source_witness=d.get('source_witness'),rows=rows,heads=q.shape[1],queries_per_head=len(sites),groups=len(costs),budget_per_head=budget,
        per_head_group_difference=[len(set(x)^set(y)) for x,y in zip(chosen['matched_sum'],chosen['query_balanced'])],
        a_tensor_bytes=a.numel()*a.element_size(),wall_s=time.perf_counter()-started,
        scope='exact bounded CPU greedy reference; own real steady input; not production selector or native numerical gate',
        limitations=['coverage .8 is proxy utility, not true Attention preservation','FP32 deletion/selection output recomputed from real KV, not utility self-validation','global output L2 is not video quality','per-head token budgets fixed; physical unions separately reported'])
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');torch.save(chosen,args.output/'selected_groups.pt');print(json.dumps(result))

if __name__=='__main__':main()
