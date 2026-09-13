#!/usr/bin/env python3
"""Actual 5B shape, complete-layout bounded KVOut R4/R8 versus native FA2."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.longlive_sparse.dataflow_reference import DataflowInputs
from adapters.longlive_sparse.bounded_dataflow_reference import BoundedKVOut
from wave2_capture_io import load_call


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for name in ('case','source','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--legacy-source',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    sys.path.insert(0,str(a.source));from wan_5b.modules.attention import attention,FLASH_ATTN_2_AVAILABLE
    if not FLASH_ATTN_2_AVAILABLE:raise RuntimeError('native FA2 required')
    data,_=load_call(a.case,legacy_source=a.legacy_source);q,k,v=[data[x].cuda() for x in ('q','k','v')]
    def run(tiles):
        if not tiles:return attention(q,k,v),0
        inputs=DataflowInputs(q[0].permute(1,0,2).contiguous(),k[0].permute(1,0,2).contiguous(),
            v[0].permute(1,0,2).contiguous(),torch.zeros(k.shape[1],device=q.device,dtype=torch.int32),3)
        state=BoundedKVOut(inputs,tiles);output=state.run().permute(1,0,2)[None].contiguous()
        return output,state.workspace_bytes
    reference,_=run(0);gates=[]
    for tiles in (4,8):
        torch.cuda.synchronize();start=time.perf_counter();out,space=run(tiles);torch.cuda.synchronize();startup=time.perf_counter()-start
        error=float((out.float()-reference.float()).norm()/reference.float().norm())
        if not torch.isfinite(out).all() or error>.02:raise RuntimeError(f'R{tiles} native full graph numerical mismatch:{error}')
        gates.append(dict(R=tiles,relative_L2=error,first_call_compile_and_allocate_s=startup,workspace_bytes=space));del out
    samples={r:[] for r in (0,4,8)};peaks={r:[] for r in samples};order=list(samples)
    for trial in range(25):
        for tiles in order[trial%3:]+order[:trial%3]:
            torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            out,space=run(tiles);torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if trial>=5:samples[tiles].append(elapsed);peaks[tiles].append(torch.cuda.max_memory_allocated()-base)
            del out
    rows=[dict(R=r,method='native_FA2' if r==0 else f'bounded_KVOut_R{r}',median_s=float(np.median(samples[r])),p95_s=float(np.quantile(samples[r],.95)),actual_extra_allocator_peak_bytes=max(peaks[r])) for r in samples]
    result=dict(status='pass',GPU=torch.cuda.get_device_name(),q_shape=list(q.shape),k_shape=list(k.shape),rows=rows,gates=gates,
        same_edges=q.shape[1]*q.shape[2]*k.shape[1],dtype=str(q.dtype),same_original_KV=True,
        included='Q/K/V layout, roles/index allocation+validation, partial/running/output allocation, all partial and merge launches, final output layout',
        common_upstream_excluded='projection, RoPE, native window/source binding already in captured input',
        input_scope=data.get('input_scope','new small steady gate input'),measured_each=20,warmup_each=5,interleaved=True,
        no_dense_fallback=True,no_HBM_counter_claim=True,no_video_speed_claim=True)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
