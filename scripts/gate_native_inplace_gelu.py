#!/usr/bin/env python3
"""Exhaustive BF16 scalar equivalence and real FFN-shape allocation gate."""
import argparse
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.longlive_sparse.native_inplace_gelu import InplaceNativeGELU


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    rows=[]
    for approximate in ('none','tanh'):
        values=torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        expected=torch.nn.functional.gelu(values,approximate=approximate)
        x=values.clone();address=x.data_ptr();out=InplaceNativeGELU(approximate).eval()(x)
        same=torch.equal(expected.view(torch.int16),out.view(torch.int16))
        if not same or out.data_ptr()!=address:raise RuntimeError('BF16 operator/output-alias gate failed')
        rows.append(dict(approximation=approximate,all65536_BF16_bit_patterns_equal=True,output_reuses_input=True))
    x=torch.randn((1,7040,14336),device='cuda',dtype=torch.bfloat16)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
    y=torch.nn.functional.gelu(x,approximate='tanh');torch.cuda.synchronize()
    normal_extra=torch.cuda.max_memory_allocated()-base
    del y
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
    y=InplaceNativeGELU('tanh').eval()(x);torch.cuda.synchronize()
    inplace_extra=torch.cuda.max_memory_allocated()-base
    report=dict(status='pass',GPU=torch.cuda.get_device_name(),rows=rows,shape=[1,7040,14336],
        normal_peak_extra_bytes=normal_extra,inplace_peak_extra_bytes=inplace_extra,
        scope='operator numerical/allocation gate; complete native video equivalence remains required')
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
