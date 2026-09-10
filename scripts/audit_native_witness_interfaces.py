#!/usr/bin/env python3
"""Separate dense FA2 API, native varlen API, and original head-count geometry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_native_attention_teacher import output_error


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'third_party/LongLive2')
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ('LLV2_USE_FA3','LLV2_USE_FA4','LLV2_USE_TE_ATTN'):os.environ[name]='0'
    sys.path.insert(0,str(args.source))
    from wan_5b.modules.attention import attention
    import flash_attn
    torch.set_num_threads(2)
    path=args.case/'numeric_witness.pt';summary=json.loads((args.case/'summary.json').read_text())
    assert summary['status']=='pass' and summary['observer_noise_latent_RGB_equivalence']
    with path.open('rb') as h:sha=hashlib.file_digest(h,'sha256').hexdigest()
    assert sha==summary['numeric_witness']['sha256']
    data=torch.load(path,map_location='cpu',weights_only=True);assert data['complete']
    rows=[]
    for r in data['records']:
        q,k,v=[r[n][None,:,None,:].cuda() for n in ('q','k','v')]
        expected=r['native_output'].cuda()
        dense=flash_attn.flash_attn_func(q,k,v,causal=False)[0,:,0]
        one=attention(q,k,v)[0,:,0]
        q24,k24,v24=[t.repeat(1,1,24,1) for t in (q,k,v)]
        full=attention(q24,k24,v24)[0,:,r['head']]
        sites=torch.tensor(r['query_indices'],device='cuda')
        row={n:r[n] for n in ('query_frame','phase','layer','head')}
        row.update(dense_FA2_one_head=output_error(expected,dense),
            native_varlen_one_head=output_error(expected,one),
            native_varlen_original_head_count=output_error(expected,full),
            native_varlen_original_head_count_sampled=output_error(expected.index_select(0,sites),full.index_select(0,sites)),
            target_head_original_inputs=True,other_heads_replicated_not_original=True)
        rows.append(row);print(json.dumps(row),flush=True)
        del q24,k24,v24,full,dense,one
    report=dict(status='diagnostic_complete',rows=rows,payload_sha256=sha,
        corrects_v1_label='same_FA2_head_replay_full_Q in v1 used dense FA2 API, not native varlen API',
        old_gate_unchanged=True,no_layer_policy_or_second_seed_released=True)
    (args.output/'interfaces.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
