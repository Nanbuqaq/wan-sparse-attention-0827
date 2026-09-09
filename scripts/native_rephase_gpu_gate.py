#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_temporal_rephase import rephase_temporal_keys
from scripts.analyze_native_attention_teacher import output_error


@torch.inference_mode()
def main():
    from flash_attn import flash_attn_func
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2)
    generator=torch.Generator().manual_seed(20260909)
    q=torch.randn(1,128,4,128,generator=generator).bfloat16()
    k=torch.randn(1,1024,4,128,generator=generator).bfloat16();v=torch.randn(1,1024,4,128,generator=generator).bfloat16()
    rows=[]
    for delta in (0,24,40,48,64):
        cpu=rephase_temporal_keys(k,delta);device=k.cuda();gpu=rephase_temporal_keys(device,delta)
        assert torch.equal(cpu,gpu.cpu()) and torch.equal(gpu[...,44:].cpu(),k[...,44:])
        if delta==0:assert gpu is device
        scores=torch.einsum('bqhd,bkhd->bhqk',q.float(),cpu.float())*128**-.5
        reference=torch.einsum('bhqk,bkhd->bqhd',scores.softmax(-1),v.float())
        actual=flash_attn_func(q.cuda(),gpu,v.cuda(),causal=False).cpu()
        error=output_error(reference,actual)
        assert error['max_abs']<=.02 and error['relative_l2']<=.01 and error['one_minus_cosine']<=.001
        rows.append(dict(delta=delta,CPU_GPU_transformed_K_bitwise_exact=True,spatial_channels_exact=True,error=error))
    result=dict(status='pass',GPU=torch.cuda.get_device_name(),rows=rows,
        preRoPE_raw_key_recovery_not_claimed=True,gate_reference_is_transformation_of_stored_BF16_K=True)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
