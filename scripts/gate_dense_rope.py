#!/usr/bin/env python3
"""Compare direct-output dense RoPE with the locked upstream, bit for bit."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.dense_rope import direct_output_causal_rope
from adapters.longlive_sparse.phase_prototypes import canonical_wan_frequency_table


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise ValueError('preserve prior gate')
    if not torch.cuda.is_available(): raise RuntimeError('real CUDA required')
    from adapters.longlive_sparse.runtime_attention import causal_online_rope
    torch.set_num_threads(2);torch.manual_seed(413)
    rows=[]
    for frames,batch,padding,strided in ((3,1,0,False),(12,1,0,False),(2,2,17,False),(3,1,0,True)):
        length=frames*1560+padding
        x=torch.randn(batch,length,12,256 if strided else 128,device='cuda',dtype=torch.bfloat16)
        if strided:x=x[...,::2]
        grid=torch.tensor([[frames,30,52]]*batch,device='cuda')
        freqs=canonical_wan_frequency_table(128).cuda()
        relative=torch.arange(frames,device='cuda').flip(0)
        values={};samples={name:[] for name in ('upstream','direct_output')}
        first={}
        for repeat in range(36):
            methods=[('upstream',causal_online_rope),('direct_output',direct_output_causal_rope)]
            if repeat%2:methods.reverse()
            for name,fn in methods:
                torch.cuda.synchronize();begin=time.perf_counter()
                output=fn(x,grid,freqs,relative_frame_indices=relative)
                torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                if repeat==0:first[name]=elapsed
                if repeat>=6:samples[name].append(elapsed)
                values[name]=output
        if not torch.equal(values['upstream'],values['direct_output']):
            raise AssertionError('same-precision RoPE changed values')
        row=dict(frames=frames,batch=batch,padding=padding,strided=strided,bitwise_equal=True,
            first_invocation_s=first,warmup=5,measurements=30,samples_s=samples,
            median_s={name:statistics.median(v) for name,v in samples.items()},
            p95_s={name:float(torch.tensor(v).quantile(.95)) for name,v in samples.items()})
        row['median_speedup']=row['median_s']['upstream']/row['median_s']['direct_output']
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='samples_s'}),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(dict(status='pass',gpu=torch.cuda.get_device_name(),rows=rows,
        video_speedup_claim=False,arithmetic_precision='FP64_complex_unchanged'),handle,indent=2);handle.write('\n')


if __name__=='__main__':main()
