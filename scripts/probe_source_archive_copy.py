"""Actual past-native KV payloads: pageable, pinned-final, bounded staging.

This is one-layer archive-copy replay, not multi-layer allocator pressure or a
video speed result. Host costs and CUDA stream spans are not added together.
"""
import argparse,hashlib,json,statistics,time
from pathlib import Path
import torch


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=7)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    torch.set_num_threads(2)
    rows=[]
    for case in args.cases:
        path=case/'steady_observer.pt';data=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
        with path.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        for c in data['calls']:
            selected=c['eligible'][:8];slots=[x[0] for x in selected]
            assert c['frame_tokens']==880 and slots==list(range(slots[0],slots[0]+8))
            assert all(o[0]=='native' and o[1]<c['frame'] for _,o in selected)
            cpu=[c[name][:,slots[0]*880:(slots[-1]+1)*880].contiguous() for name in ('k','v')]
            gpu=[x.cuda() for x in cpu];torch.cuda.synchronize()
            bytes_=sum(x.numel()*x.element_size() for x in gpu)
            start=time.perf_counter();stage=torch.empty_like(cpu[0],pin_memory=True)
            stage_setup_ms=1000*(time.perf_counter()-start)
            def copy(method):
                outputs=[];copy_ms=0.;CPU_copy_ms=0.;begin=time.perf_counter()
                for src in gpu:
                    a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)];a.record()
                    if method=='pageable':
                        out=src.detach().to('cpu',copy=True).contiguous();b.record();b.synchronize()
                    elif method=='pinned_final':
                        out=torch.empty_like(src,device='cpu',pin_memory=True)
                        out.copy_(src,non_blocking=True);b.record();b.synchronize()
                    else:
                        stage.copy_(src,non_blocking=True);b.record();b.synchronize()
                        t=time.perf_counter();out=torch.empty_like(stage,pin_memory=False);out.copy_(stage)
                        CPU_copy_ms+=1000*(time.perf_counter()-t)
                    copy_ms+=a.elapsed_time(b);outputs.append(out)
                return outputs,dict(host_complete_ms=1000*(time.perf_counter()-begin),
                    D2H_stream_span_ms_including_submission_gaps=copy_ms,extra_CPU_allocate_and_copy_ms=CPU_copy_ms)
            names=('pageable','pinned_final','pinned_stage');samples={x:[] for x in names};cold={}
            for rep in range(args.repeats+1):
                order=list(names);order=order[rep%3:]+order[:rep%3]
                for method in order:
                    torch.cuda.synchronize();out,timing=copy(method)
                    assert all(torch.equal(a,b) for a,b in zip(cpu,out)),method
                    assert all(x.is_pinned()==(method=='pinned_final') for x in out)
                    if rep:samples[method].append(timing)
                    else:cold[method]=timing
                    del out
            traces={}
            for method in names:
                trace=args.output/f'{case.name}_{c["frame"]}_{method}.json'
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    with torch.profiler.record_function('archive_copy_'+method):out,_=copy(method)
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(trace));del out
                events=json.loads(trace.read_text())['traceEvents']
                dma=[e for e in events if e.get('ph')=='X' and 'DtoH' in e.get('name','') and e.get('cat')=='gpu_memcpy']
                traces[method]=dict(path=str(trace),D2H_CUPTI_service_ms=sum(e['dur'] for e in dma)/1000,
                    observed_DMA_events=len(dma),not_GPU_SM_utilization=True)
            row=dict(case=case.name,capture_sha256=sha,frame=c['frame'],layer=c['layer'],source_owners=[o for _,o in selected],
                source_shapes=[list(x.shape) for x in cpu],actual_D2H_bytes=bytes_,setup_H2D_bytes=bytes_,
                staged_extra_CPU_copy_bytes=bytes_,pinned_stage_owned_bytes=stage.numel()*stage.element_size(),
                pinned_final_owned_bytes=bytes_,stage_setup_ms=stage_setup_ms,bitwise_copy_correct=True,
                cold=cold,samples=samples,median={m:{k:statistics.median(s[k] for s in ss) for k in ss[0]} for m,ss in samples.items()},traces=traces)
            rows.append(row);print(json.dumps({k:row[k] for k in ('case','frame','median','traces')}),flush=True)
    with (args.output/'result.json').open('x') as f:json.dump(dict(status='pass',GPU=torch.cuda.get_device_name(),rows=rows,
        scope='one-layer actual native-past payload replay; not full archive allocator pressure, overlap or delivery proof',
        warm_pinned_allocator_reuse_not_free_initial_archive_allocation=True),f,indent=2)

if __name__=='__main__':main()
