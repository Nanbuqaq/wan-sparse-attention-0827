#!/usr/bin/env python3
"""Independent direct Attention, group reconstruction, and actual BF16 replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_native_attention_teacher import output_error


@torch.inference_mode()
def main():
    import flash_attn
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('real GPU required for original BF16 backend replay')
    summary=json.loads((args.case/'summary.json').read_text())
    assert summary['status']=='pass' and summary['observer_noise_latent_RGB_equivalence']
    path=args.case/'numeric_witness.pt'
    with path.open('rb') as h:sha=hashlib.file_digest(h,'sha256').hexdigest()
    assert sha==summary['numeric_witness']['sha256']
    data=torch.load(path,map_location='cpu',weights_only=True);assert data['complete']
    torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    for r in data['records']:
        q,k,v=[r[name].cuda() for name in ('q','k','v')]
        sites=torch.tensor(r['query_indices'],device='cuda',dtype=torch.long)
        qs=q.index_select(0,sites)
        ref=((qs.float()@k.float().T)*(128**-.5)).softmax(-1)@v.float()
        exact64=((qs.double()@k.double().T)*(128**-.5)).softmax(-1)@v.double()
        z=[];os=[]
        for a in range(0,k.shape[0],8*r['frame_tokens']):
            kk=k[a:a+8*r['frame_tokens']].float();vv=v[a:a+8*r['frame_tokens']].float()
            logits=(qs.float()@kk.T)*(128**-.5)
            z.append(logits.logsumexp(-1));os.append(logits.softmax(-1)@vv)
        z=torch.stack(z,-1);os=torch.stack(os,-2)
        merged=(z.softmax(-1)[...,None]*os).sum(-2)
        replay=flash_attn.flash_attn_func(q[None,:,None,:],k[None,:,None,:],v[None,:,None,:],causal=False)[0,:,0]
        original=r['native_output'].cuda();sampled=original.index_select(0,sites)
        row={key:r[key] for key in ('query_frame','phase','layer','head')}
        row.update(direct_FP32_vs_native=output_error(ref,sampled),
            direct_FP64_vs_FP32=output_error(exact64,ref),
            group_FP32_vs_direct_FP32=output_error(ref,merged),
            native_vs_same_FA2_head_replay_full_Q=output_error(original,replay),
            nearest_BF16_rounding_max_abs=float((ref-ref.bfloat16().float()).abs().max()),
            original_absolute_threshold=.02)
        rows.append(row)
        print(json.dumps(row),flush=True)
    report=dict(status='diagnosis_complete_original_gate_unchanged',gpu=torch.cuda.get_device_name(),
                payload_sha256=sha,full_generator_output_matches_reference=True,rows=rows,
                no_second_seed_or_layer_policy_released=True)
    (args.output/'numeric_diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
