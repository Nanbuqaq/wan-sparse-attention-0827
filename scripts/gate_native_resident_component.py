#!/usr/bin/env python3
"""Real native-shape GPU gate with an independent, renormalized FP32 teacher."""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_resident_history import summarize_frame,contrast_scores,choose_whole_blocks
from scripts.analyze_native_attention_teacher import output_error


@torch.inference_mode()
def main():
    import flash_attn
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--required-gpu-name',default='');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():raise RuntimeError('real CUDA required')
    if args.required_gpu_name and args.required_gpu_name not in torch.cuda.get_device_name():
        raise RuntimeError('assigned GPU does not match frozen hardware identity')
    torch.set_num_threads(2);torch.manual_seed(20260910)
    torch.backends.cuda.matmul.allow_tf32=False
    device='cuda';dtype=torch.bfloat16
    q=torch.randn(1,7040,24,128,device=device,dtype=dtype)*.25
    k=torch.randn(1,7040,24,128,device=device,dtype=dtype)*.25
    v=torch.randn_like(k)
    frames=[summarize_frame(k[0,i:i+880],v[0,i:i+880]) for i in range(0,7040,880)]
    km,vm,count=[torch.cat([row[j] for row in frames]) for j in range(3)]
    rows=[]
    for method in ('mass_value','contrast_value'):
        scores=contrast_scores(q[0],km,vm,count,method).cpu().tolist()
        selected,n,budget=choose_whole_blocks(scores,count.cpu().tolist(),.25)
        ids=[]
        for b in selected:
            frame,block=divmod(b,14);start=frame*880+block*64
            ids.extend(range(start,start+min(64,880-block*64)))
        ids=torch.tensor(ids,device=device)
        sk,sv=k.index_select(1,ids),v.index_select(1,ids)
        actual=flash_attn.flash_attn_func(q,sk,sv,causal=False)
        query_ids=torch.linspace(0,7039,32,device=device).round().long()
        qq=q.index_select(1,query_ids)[0].permute(1,0,2).float()
        kk=sk[0].permute(1,0,2).float();vv=sv[0].permute(1,0,2).float()
        # Direct teacher normalizes on the actual remaining original KV.
        ref=(qq@kk.transpose(-1,-2)/(128**.5)).softmax(-1)@vv
        observed=actual.index_select(1,query_ids)[0].permute(1,0,2)
        error=output_error(ref,observed)
        passed=error['max_abs']<=.02 and error['relative_l2']<=.01 and error['one_minus_cosine']<=.001
        rows.append(dict(method=method,status='pass' if passed else 'fail',error=error,
            selected_tokens=n,budget_tokens=budget,candidate_tokens=7040,
            actual_Q=7040,actual_K=n,heads=24,head_dim=128,teacher_queries_per_head=32,
            independent_teacher='direct_FP32_on_remaining_original_KV_with_fresh_softmax',
            native_backend='flash_attn_func',real_GPU=True,synthetic_operator_gate_not_video_quality=True))
    report=dict(status='pass' if all(r['status']=='pass' for r in rows) else 'fail',
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,rows=rows,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated())
    (args.output/'gate.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
