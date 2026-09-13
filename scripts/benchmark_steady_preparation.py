#!/usr/bin/env python3
"""Fixed real steady routes: potential kernel savings, pack tax and exact ablations."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.longlive_sparse.query_balanced_value import stratified_sites,normalized_values,select_batched,select_static_once,execute_per_head


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for name in ('case','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frame',type=int,choices=(24,88),required=True);p.add_argument('--profile',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    sys.path.insert(0,str(a.source));from wan_5b.modules.attention import attention,FLASH_ATTN_2_AVAILABLE
    from flash_attn import flash_attn_varlen_func
    if not FLASH_ATTN_2_AVAILABLE:raise RuntimeError('native FA2 required')
    payload=torch.load(a.case/'steady_observer.pt',weights_only=True,map_location='cpu',mmap=True)
    d=next(x for x in payload['calls'] if x['frame']==a.frame)
    q,k,v,km,vm,counts=[d[x].cuda() for x in ('q','k','v','key_mean','value_mean','counts')]
    chosen=d['selected_group_mask'].cuda();costs=d['counts'].tolist();budget=sum(costs)//2;upper=len(d['protected'])+budget
    heads,length,dim=q.shape[2],q.shape[1],q.shape[3];perframe=(d['frame_tokens']+63)//64
    def geometry():
        mapping=torch.full((k.shape[1],),-2,dtype=torch.long);mapping[d['protected']]=-1
        for group,indices in enumerate(d['token_blocks']):mapping[indices]=group
        if (mapping==-2).any():raise ValueError('incomplete physical mapping')
        return mapping.cuda(),stratified_sites(8,*d['token_grid']).cuda()
    mapping,sites=geometry()
    visible=chosen[:,mapping.clamp_min(0)]|(mapping[None]<0);coords=visible.nonzero()
    hq=q[0].permute(1,0,2).contiguous().reshape(heads*length,1,dim)
    hk=k[0].permute(1,0,2)[coords[:,0],coords[:,1]][:,None];hv=v[0].permute(1,0,2)[coords[:,0],coords[:,1]][:,None]
    cuq=torch.arange(heads+1,device=q.device,dtype=torch.int32)*length
    cuk=torch.cat([torch.zeros(1,device=q.device,dtype=torch.int32),visible.sum(1,dtype=torch.int32).cumsum(0,dtype=torch.int32)])
    aa=normalized_values(q[0,sites],km,vm,counts)
    old=select_batched(aa,costs,budget,balanced=False)[0];fast=select_static_once(aa,costs,budget)[0]
    if not torch.equal(old,chosen) or not torch.equal(fast,chosen):raise RuntimeError('actual per-head route differs on matched input')
    reproduced,_=execute_per_head(q,k,v,chosen,mapping,upper)
    source_report=json.loads((a.case/'summary.json').read_text());captured_equal=torch.equal(reproduced.cpu(),d['output'])
    if not captured_equal and source_report['gpu']==torch.cuda.get_device_name():
        raise RuntimeError('same-platform native per-head output differs')
    # Cross-architecture replay has its own FP32 numerical witness; never claim
    # cross-GPU bitwise equivalence or substitute its timing for the video GPU.
    torch.backends.cuda.matmul.allow_tf32=False
    numerator=0.;captured_numerator=0.;denominator=0.
    for head in range(heads):
        selected_k=visible[head];qq=q[0,sites,head].float();kk=k[0,selected_k,head].float();vv=v[0,selected_k,head].float()
        teacher=(qq@kk.T/dim**.5).softmax(-1)@vv
        numerator+=float((reproduced[0,sites,head].float()-teacher).square().sum())
        captured_numerator+=float((d['output'][0,sites.cpu(),head].cuda().float()-teacher).square().sum())
        denominator+=float(teacher.square().sum())
    local_error=(numerator/denominator)**.5;captured_error=(captured_numerator/denominator)**.5
    if max(local_error,captured_error)>.02:raise RuntimeError('independent FP32 selected-output numerical gate failed')
    def run(name):
        if name=='native_direct':return attention(q,k,v)
        if name=='prepared_selected_attention':return flash_attn_varlen_func(hq,hk,hv,cuq,cuk,length,upper,dropout_p=0.,causal=False)
        if name=='fixed_route_pack_attention':return execute_per_head(q,k,v,chosen,mapping,upper)[0]
        mm,ss=(mapping,sites) if name=='geometry_cache' else geometry()
        # Both paths update bound prototypes on this first-denoise input.
        keys,values,cc=[torch.cat([t[i:i+perframe] for i in range(0,len(costs),perframe)]) for t in (km,vm,counts)]
        values_a=normalized_values(q[0,ss],keys,values,cc)
        selected,coverage,used=(select_batched(values_a,costs,budget,balanced=False) if name=='online_old' else select_static_once(values_a,costs,budget))
        stats=torch.cat([used.float(),coverage.amin(1),coverage.mean(1)])
        if name in ('online_old','static_sort'):stats.cpu()
        output,vis=execute_per_head(q,k,v,selected,mm,upper);union=vis.any(0).sum()
        if name in ('online_old','static_sort'):union.cpu()
        else:torch.cat([stats,union.float()[None]]).cpu()
        return output
    names=['native_direct','prepared_selected_attention','fixed_route_pack_attention','online_old','static_sort','deferred_stats','geometry_cache']
    times={n:[] for n in names};peaks={n:[] for n in names}
    for trial in range(25):
        for name in names[trial%len(names):]+names[:trial%len(names)]:
            torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            out=run(name);torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if trial>=5:times[name].append(elapsed);peaks[name].append(torch.cuda.max_memory_allocated()-base)
            del out
    profiles={}
    if a.profile:
        for name in names:
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    out=run(name);torch.cuda.synchronize()
                path=a.output/(name+'_trace.json');prof.export_chrome_trace(str(path));events=json.loads(path.read_text())['traceEvents']
                kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
                if not kernels:raise RuntimeError('CUPTI returned no CUDA kernel records')
                sync=[e for e in events if e.get('cat')=='cuda_runtime' and 'Synchronize' in e.get('name','')]
                profiles[name]=dict(kernel_count=len(kernels),kernel_service_s=sum(e['dur'] for e in kernels)/1e6,host_sync_calls=len(sync),host_sync_span_s=sum(e.get('dur',0) for e in sync)/1e6)
            except Exception as e:profiles[name]=dict(status='profile_unavailable',error=repr(e))
    report=dict(status='pass',frame=a.frame,source=str(a.case),GPU=torch.cuda.get_device_name(),q_shape=list(q.shape),k_shape=list(k.shape),groups=len(costs),original_optional_budget_per_head=budget,
        complete_actual_route_equal=True,captured_output_bitwise_equal=captured_equal,source_GPU=source_report['gpu'],
        independent_FP32_local_relative_L2=local_error,independent_FP32_captured_relative_L2=captured_error,
        rows=[dict(method=n,median_s=float(np.median(times[n])),p95_s=float(np.quantile(times[n],.95)),extra_peak_bytes=max(peaks[n])) for n in names],profiles=profiles,
        measured_each=20,warmup_each=5,interleaved=True,geometry_H2D_bytes_per_rebuild=mapping.numel()*mapping.element_size()+sites.numel()*sites.element_size(),
        scope='first-denoise fixed real input; scopes deliberately separated, no addition into video time',
        deferred_scope='conservative one-call accounting includes final stats readback; cross-layer chunk batching benefit is measured in whole videos',
        common_excluded='clean summary production, projections/RoPE/window; bound summary cat included in online scopes',
        native_direct_has_no_new_layout_tax=True,prepared_attention_output_layout='head-as-batch flat output; final model-layout conversion excluded only in this kernel-potential scope')
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
