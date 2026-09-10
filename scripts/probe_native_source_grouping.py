#!/usr/bin/env python3
"""B-line grouping diagnostic on existing real native QKV; not a video claim."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.native_resident_history import contrast_scores
from scripts.analyze_native_attention_teacher import output_error


def source_groups(kind,frames=8,height=22,width=40):
    tokens=height*width;groups=[]
    for f in range(frames):
        base=f*tokens
        if kind=='flat64':
            groups.extend([list(range(base+a,base+min(a+64,tokens))) for a in range(0,tokens,64)])
        elif kind=='spatial8':
            for y in range(0,height,8):
                for x in range(0,width,8):
                    groups.append([base+yy*width+xx for yy in range(y,min(y+8,height)) for xx in range(x,min(x+8,width))])
        elif kind=='flat_matched':
            lengths=[len(g) for g in source_groups('spatial8',1,height,width)];begin=base
            for length in lengths:groups.append(list(range(begin,begin+length)));begin+=length
        else:raise ValueError('unknown partition')
    if sorted(i for g in groups for i in g)!=list(range(frames*tokens)):
        raise ValueError('partition does not exactly cover the source')
    return groups


def ranked_exact_tokens(groups,scores,budget):
    # Analysis control only: trim the final group to compare EXACT raw budgets.
    # Production whole-group admission is a different recorded policy.
    order=sorted(range(len(groups)),key=lambda i:(-scores[i],i))
    return sorted([t for i in order for t in groups[i]][:budget])


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    torch.backends.cuda.matmul.allow_tf32=False
    with args.capture.open('rb') as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
    data=torch.load(args.capture,map_location='cpu',weights_only=True,mmap=True)
    if not data.get('online_routing_may_not_access'):raise ValueError('offline boundary flag missing')
    rows=[]
    for ri,record in enumerate(data['records']):
        assert record['frame_tokens']==880 and record['query_frame']==96 and record['q'].shape[1]==32
        q,k,v=[record[n].cuda() for n in ('q','k','v')]
        assert k.shape[1]==28160 and q.shape[2:]==(24,128)
        qh=q[0].permute(1,0,2).float().contiguous()
        kh=k[0].permute(1,0,2).float().contiguous();vh=v[0].permute(1,0,2).float().contiguous()
        full_mass=(qh@kh.transpose(-1,-2)*(128**-.5)).softmax(-1)
        full=full_mass@vh
        native=record['native_output'][0].permute(1,0,2).cuda()
        native_error=output_error(full,native)
        gate=native_error['max_abs']<=.02 and native_error['relative_l2']<=.01 and native_error['one_minus_cosine']<=.001
        source_start=7040;source_tokens=7040;budget=1760
        exact=list(range(source_start))+list(range(source_start+source_tokens,k.shape[1]))
        all_source_mass=full_mass[:,:,source_start:source_start+source_tokens].sum(-1)
        variants=[]
        for kind in ('flat64','flat_matched','spatial8'):
            groups=source_groups(kind)
            means=[]
            for tensor in (k,v):
                means.append(torch.stack([tensor[0].index_select(0,torch.tensor([source_start+i for i in g],device='cuda')).float().mean(0) for g in groups]))
            counts=torch.tensor([len(g) for g in groups],device='cuda')
            for policy in ('mass_value','contrast_value'):
                scores=contrast_scores(q[0],means[0],means[1],counts,policy,samples=32).cpu().tolist()
                selected=ranked_exact_tokens(groups,scores,budget)
                variants.append((kind,policy,selected,len(groups)))
            del means,counts
        random_ids=sorted(random.Random(20260910+ri).sample(range(source_tokens),budget))
        variants.append(('token_random','random',random_ids,source_tokens))
        for kind,policy,selected,groups_count in variants:
            ids=torch.tensor(exact+[source_start+i for i in selected],device='cuda').sort().values
            remaining_k=kh.index_select(1,ids);remaining_v=vh.index_select(1,ids)
            output=(qh@remaining_k.transpose(-1,-2)*(128**-.5)).softmax(-1)@remaining_v
            kept_mass=full_mass.index_select(-1,torch.tensor([source_start+i for i in selected],device='cuda')).sum(-1)
            row=dict(capture_row=ri,layer=record['layer'],phase=record['phase'],grouping=kind,policy=policy,
                groups=groups_count,selected_source_tokens=len(selected),source_budget=budget,
                protected_tokens=len(exact),native_reference_gate=gate,native_reference_error=native_error,
                selected_output_error=output_error(full,output),
                source_mass_retained_fraction=float((kept_mass/all_source_mass.clamp_min(1e-12)).mean()))
            rows.append(row);print(json.dumps({k:row[k] for k in ('capture_row','grouping','policy','native_reference_gate','selected_output_error')}),flush=True)
        del q,k,v,qh,kh,vh,full_mass,full,remaining_k,remaining_v,output
    report=dict(status='offline_diagnostic_complete',rows=rows,capture_sha256=digest,
        sampled_Q_only=True,full_Q_policy_or_video_quality_claimed=False,
        representative_sites='existing geometric sample; not the live uniform32 policy',
        source='existing related_recent trajectory, source group is second8-frame segment',
        budget_control='exact1760 original source tokens; final-group trimming is offline-only',
        grouping_size_control='flat_matched and spatial8 have identical group counts and sizes',
        no_captured_tensor_is_an_online_router_input=True,
        no_failed_numeric_row_may_select_a_layer_policy=True)
    (args.output/'grouping_probe.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
