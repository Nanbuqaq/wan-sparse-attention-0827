#!/usr/bin/env python3
"""Interleaved native/legacy/cached selection component on one actual 5B window."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_resident_history import contrast_scores,choose_whole_blocks


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for name in ('case','source','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    sys.path.insert(0,str(args.source));from wan_5b.modules.attention import attention,FLASH_ATTN_2_AVAILABLE
    if not FLASH_ATTN_2_AVAILABLE:raise RuntimeError('native FA2 required')
    d=torch.load(args.case/'wave2_diagnostics.pt',weights_only=True,map_location='cpu')['call']
    q,k,v=[d[name].to('cuda') for name in ('q','k','v')]
    km,vm,count=[d[name].to('cuda') for name in ('key_mean','value_mean','counts')]
    groups=d['token_blocks'];sizes=[len(g) for g in groups];ft=d['frame_tokens'];per_frame=(ft+63)//64
    frame_summaries=[(km[a:a+per_frame],vm[a:a+per_frame],count[a:a+per_frame]) for a in range(0,len(groups),per_frame)]
    protected_frames=sorted({x//ft for x in d['protected']});timings={name:[] for name in ('native_direct','existing_resident','cached_descriptors_summaries')};peaks={name:[] for name in timings}
    def run(name):
        if name=='native_direct':return attention(q,k,v),None
        if name=='existing_resident':
            blocks=[list(range(position*ft+start,position*ft+min(start+64,ft))) for position,_ in d['eligible'] for start in range(0,ft,64)]
            costs=[len(x) for x in blocks];protected=[x for frame in protected_frames for x in range(frame*ft,(frame+1)*ft)]
            keys=torch.cat([x[0] for x in frame_summaries]);values=torch.cat([x[1] for x in frame_summaries]);counts=torch.cat([x[2] for x in frame_summaries])
        else:blocks,costs,protected,keys,values,counts=groups,sizes,d['protected'],km,vm,count
        scores=contrast_scores(q[0],keys,values,counts,'mass_value',32).cpu().tolist()
        selected,_,_=choose_whole_blocks(scores,costs,.5)
        indices=torch.tensor(sorted(protected+[t for b in selected for t in blocks[b]]),device='cuda',dtype=torch.long)
        return attention(q,k.index_select(1,indices),v.index_select(1,indices)),indices
    old,old_idx=run('existing_resident');new,new_idx=run('cached_descriptors_summaries')
    if not torch.equal(old_idx,new_idx) or not torch.equal(old,new):raise RuntimeError('same-input route/output gate failed')
    if not torch.equal(new_idx.cpu(),d['selected_indices']):raise RuntimeError('replay differs from actual stored route')
    names=list(timings)
    for trial in range(25):
        for name in names[trial%3:]+names[:trial%3]:
            torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
            result,_=run(name);torch.cuda.synchronize();elapsed=time.perf_counter()-began
            if trial>=5:timings[name].append(elapsed);peaks[name].append(torch.cuda.max_memory_allocated()-baseline)
            del result
    rows=[dict(method=name,median_s=float(np.median(times)),p95_s=float(np.quantile(times,.95)),samples=len(times),extra_allocator_peak_bytes=max(peaks[name])) for name,times in timings.items()]
    report=dict(status='pass',GPU=torch.cuda.get_device_name(),shape=dict(q=list(q.shape),k=list(k.shape)),rows=rows,
        exact_same_route_and_output=True,warmup_each=5,measured_each=20,interleaved=True,
        native_K=k.shape[1],selected_K=new_idx.numel(),
        scope='prepared native Q/K/V window; native path has no conversion tax; sparse paths include score/sync/indexH2D/gather/FA2',
        excluded_common_upstream='projections/RoPE/native window construction and clean-summary production; measured separately in video/profile',
        no_layer_multiplication_into_video_time=True,hardware_bandwidth_counters_not_measured=True)
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
