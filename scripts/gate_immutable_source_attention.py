#!/usr/bin/env python3
"""Independent FP32 and same-graph FA2 gates for the bounded side-bank merge."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.longlive_sparse.immutable_source_reader import side_attention


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    torch.set_num_threads(2);torch.manual_seed(31415);torch.backends.cuda.matmul.allow_tf32=False
    from flash_attn import flash_attn_func
    records=[]
    for qlen,native_len,source_len in ((512,4096,1024),(7040,28160,7040)):
        q=torch.randn(1,qlen,24,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(1,native_len,24,128,device='cuda',dtype=torch.bfloat16)
        v=torch.randn_like(k);sk=torch.randn(1,source_len,24,128,device='cuda',dtype=torch.bfloat16);sv=torch.randn_like(sk)
        before=[t.clone() for t in (q,k,v,sk,sv)]
        out,mass=side_attention(q,k,v,sk,sv)
        assert all(torch.equal(a,b) for a,b in zip(before,(q,k,v,sk,sv)))
        del before
        fullk=torch.cat([k,sk],dim=1);fullv=torch.cat([v,sv],dim=1)
        reference=flash_attn_func(q,fullk,fullv,dropout_p=0.,causal=False)
        sites=torch.linspace(0,qlen-1,32,device=q.device).round().long()
        qq=q[0,sites].float().permute(1,0,2)
        teacher=(((qq@fullk[0].float().permute(1,2,0))/128**.5).softmax(-1)@fullv[0].float().permute(1,0,2)).permute(1,0,2)
        rel=float(((out[0,sites].float()-teacher).square().sum()/teacher.square().sum()).sqrt())
        maxerr=float((out[0,sites].float()-teacher).abs().max())
        same=float(((out.float()-reference.float()).square().sum()/reference.float().square().sum()).sqrt())
        if rel>.01 or maxerr>.02 or same>.01:raise RuntimeError('side-bank numerical gate failed')
        records.append(dict(q_shape=list(q.shape),native_K=native_len,source_K=source_len,
            FP32_sample_relative_L2=rel,FP32_sample_max_abs=maxerr,full_Q_concat_relative_L2=same,
            source_mass_mean=float(mass.mean()),all_input_tensors_bitwise_unchanged=True))
    result=dict(status='pass',GPU=torch.cuda.get_device_name(),records=records,
        fallback=False,scope='synthetic numerical/large-shape gate only; actual source video gate remains required',
        module_sha256=hashlib.sha256((Path(__file__).resolve().parents[1]/'adapters/longlive_sparse/immutable_source_reader.py').read_bytes()).hexdigest())
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
