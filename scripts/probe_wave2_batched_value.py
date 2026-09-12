#!/usr/bin/env python3
"""New bounded GPU selector + original-KV head execution on a real captured call."""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.longlive_sparse.query_balanced_value import (
    stratified_sites, normalized_values, select_reference, select_batched, execute_per_head)
from wave2_capture_io import load_call


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for key in ('case','source','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--legacy-source',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    sys.path.insert(0,str(args.source))
    from wan_5b.modules.attention import attention, FLASH_ATTN_2_AVAILABLE
    if not FLASH_ATTN_2_AVAILABLE:raise RuntimeError('real native FA2 required')
    d,summary=load_call(args.case,legacy_source=args.legacy_source)
    q,k,v=[d[x].cuda() for x in ('q','k','v')]
    km,vm,counts=[d[x].cuda() for x in ('key_mean','value_mean','counts')]
    costs=d['counts'].long().tolist();budget=math.floor(sum(costs)*.5)
    height,width=[x//2 for x in summary['latent_shape'][-2:]]
    sites=stratified_sites(8,height,width).cuda()
    mapping=torch.full((k.shape[1],),-2,dtype=torch.long)
    mapping[d['protected']]=-1
    for b,indices in enumerate(d['token_blocks']):mapping[indices]=b
    if (mapping==-2).any():raise RuntimeError('group map does not cover original window')
    mapping=mapping.cuda();upper=len(d['protected'])+budget
    a=normalized_values(q[0,sites],km,vm,counts)
    masks={};statistics=[]
    for balanced in (False,True):
        name='balanced_batch4' if balanced else 'matched_sum_batch4'
        selected,coverage,used=select_batched(a,costs,budget,balanced=balanced)
        output,visible=execute_per_head(q,k,v,selected,mapping,upper)
        cpu=a.cpu().numpy();exact=[]
        for head in cpu:exact.append(select_reference(head,costs,budget,balanced=balanced)[0])
        actual=[row.nonzero().flatten().tolist() for row in selected.cpu()]
        # Independent FP32 full/selected Attention on all fixed representative Q.
        rel_num=0.;rel_den=0.;native_num=0.;native_den=0.
        for h in range(q.shape[2]):
            qq=q[0,sites,h].float();kk=k[0,:,h].float();vv=v[0,:,h].float()
            full=(qq@kk.T/math.sqrt(q.shape[-1])).softmax(-1)@vv
            chosen_k=visible[h];part=(qq@kk[chosen_k].T/math.sqrt(q.shape[-1])).softmax(-1)@vv[chosen_k]
            rel_num+=float((part-full).square().sum());rel_den+=float(full.square().sum())
            native_num+=float((output[0,sites,h].float()-part).square().sum());native_den+=float(part.square().sum())
        kernel_error=math.sqrt(native_num/native_den)
        if kernel_error>.02:raise RuntimeError(f'FA2 independent numerical gate failed: {kernel_error}')
        masks[name]=selected.cpu()
        statistics.append(dict(method=name,used_tokens_per_head=used.tolist(),
            physical_union=int(visible.any(0).sum()),logical_head_tokens=int(visible.sum()),
            coverage_min=float(coverage.min()),coverage_mean=float(coverage.mean()),
            groups_different_from_exact=[len(set(x)^set(y)) for x,y in zip(actual,exact)],
            FP32_selected_vs_full_relL2=math.sqrt(rel_num/rel_den),FA2_vs_FP32_selected_relL2=kernel_error))
    def run(name):
        if name=='native_direct':return attention(q,k,v)
        aa=normalized_values(q[0,sites],km,vm,counts)
        chosen,_,_=select_batched(aa,costs,budget,balanced=name=='balanced_batch4')
        return execute_per_head(q,k,v,chosen,mapping,upper)[0]
    names=['native_direct','matched_sum_batch4','balanced_batch4'];times={n:[] for n in names};peaks={n:[] for n in names}
    for iteration in range(25):
        for name in names[iteration%3:]+names[:iteration%3]:
            torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            out=run(name);torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if iteration>=5:times[name].append(elapsed);peaks[name].append(torch.cuda.max_memory_allocated()-base)
            del out
    report=dict(status='pass',input_scope=d.get('input_scope','new steady input'),source_witness=d.get('source_witness'),GPU=torch.cuda.get_device_name(),q_shape=list(q.shape),k_shape=list(k.shape),
        rows=statistics,timing=[dict(method=n,median_s=float(np.median(times[n])),p95_s=float(np.quantile(times[n],.95)),extra_peak_bytes=max(peaks[n])) for n in names],
        groups=len(costs),budget_per_head=budget,queries_per_head=len(sites),batch_size=4,cap=.8,
        scope='real captured input; BF16 FA2 original-KV per-head execution; no closed-loop quality claim',
        timing_includes='Q sampling, normalized value score, all greedy rounds, dynamic index sync, pack, FA2 and output layout',
        common_setup_excluded='existing clean summaries, immutable token-group map; must be charged on production invalidation',
        measured_each=20,warmup_each=5,approximate_not_exact_greedy=True)
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save(masks,args.output/'selected_groups.pt');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
