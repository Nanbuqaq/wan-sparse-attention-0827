#!/usr/bin/env python3
"""Bounded new 5B actual-shape dataflow test, including layout/workspace costs."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.longlive_sparse.dataflow_reference import DataflowInputs,PreparedDataflow
from wave2_capture_io import load_call


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for key in ('case','source','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--legacy-source',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    sys.path.insert(0,str(args.source));from wan_5b.modules.attention import attention,FLASH_ATTN_2_AVAILABLE
    if not FLASH_ATTN_2_AVAILABLE:raise RuntimeError('native FA2 required')
    d,_=load_call(args.case,legacy_source=args.legacy_source)
    q,k,v=[d[key].cuda() for key in ('q','k','v')]
    h,n,dim=q.shape[2],q.shape[1],q.shape[3];nt=(k.shape[1]+63)//64
    workspace=4*h*nt*n*(dim+1)
    free,total=torch.cuda.mem_get_info();kv_allowed=workspace+1024**3<free*.8
    # Actual full route is common to all queries: reference reuse=3 gives the
    # same complete graph. No synthetic tags or enlarged Q/K are introduced.
    def prepare():
        return PreparedDataflow(DataflowInputs(q[0].permute(1,0,2).contiguous(),
            k[0].permute(1,0,2).contiguous(),v[0].permute(1,0,2).contiguous(),
            torch.zeros(k.shape[1],device=q.device,dtype=torch.int32),3))
    def run(name):
        if name=='native_direct':return attention(q,k,v)
        state=prepare();out,_=state.qout() if name=='q_major_complete' else state.kvout()
        return out.permute(1,0,2)[None].contiguous()
    native=run('native_direct');names=['native_direct','q_major_complete']+(['kv_major_complete'] if kv_allowed else [])
    gates=[]
    for name in names[1:]:
        out=run(name);err=float((out.float()-native.float()).norm()/native.float().norm())
        gates.append(dict(method=name,native_relative_L2=err,status='pass' if err<.02 else 'fail'))
        if err>=.02:raise RuntimeError(f'{name} full route numerical gate failed: {err}')
        del out
    samples={n:[] for n in names};peaks={n:[] for n in names}
    for trial in range(25):
        for name in names[trial%len(names):]+names[:trial%len(names)]:
            torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            out=run(name);torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if trial>=5:samples[name].append(elapsed);peaks[name].append(torch.cuda.max_memory_allocated()-base)
            del out
    report=dict(status='pass',GPU=torch.cuda.get_device_name(),source_witness=d.get('source_witness'),input_scope=d.get('input_scope','new steady full-window input'),
        q_shape=list(q.shape),k_shape=list(k.shape),same_edges=h*n*k.shape[1],dtype=str(q.dtype),gates=gates,
        rows=[dict(method=name,median_s=float(np.median(samples[name])),p95_s=float(np.quantile(samples[name],.95)),extra_peak_bytes=max(peaks[name])) for name in names],
        kv_workspace_required_bytes=workspace,device_total_bytes=total,device_free_before_bytes=free,
        kv_status='measured' if kv_allowed else 'capacity_gate_skip_without_allocation',
        kv_limit='reference stores per-tile FP32 partial output, not a tuned KVOut lower bound',
        measured_each=20,warmup_each=5,interleaved=True,
        included='head-major Q/K/V transforms, role index allocation/validation, output/state allocation, all kernels, merge and final output layout',
        excluded_common='upstream projection, RoPE and source installation already completed in this real captured input',
        storage_scope='full GPU-resident only; no invented finite-resident H2D scenario',
        no_HBM_counter_claim=True,no_video_speed_claim=True)
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
