#!/usr/bin/env python3
"""Real forward gate: causal phase index, original KV and full FP32 Attention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.history_cache import HistoryUnionCache
from adapters.longlive_sparse.memory_dynamics import compare_coordinates
from adapters.longlive_sparse.offload import ArchiveOffloadStager
from adapters.longlive_sparse.offline_eval import routed_history_attention, output_error_metrics
from adapters.longlive_sparse.phase_prototypes import canonical_wan_frequency_table, archive_rope0_key
from adapters.longlive_sparse.selectors import summarize_query_for_pretransfer
from adapters.longlive_sparse.staging import PinnedStagingPool
from adapters.longlive_sparse.system_config import LongLiveSystemConfig


@torch.inference_mode()
def run(large):
    import adapters.longlive_sparse.runtime_attention as runtime
    torch.manual_seed(20260906)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    height,width,heads,dim,local,history,new=(30,52,12,128,12,6,3) if large else (8,16,2,64,6,2,1)
    tokens=height*width
    current=(history+local)*tokens
    freqs=canonical_wan_frequency_table(dim).cuda()
    cfg=SparseHistoryConfig(method='rope_aligned_final_history',history_density=.25,refresh_policy='per_chunk')
    archive=HistoryArchive(cfg,spatial_height=height,spatial_width=width)
    params={'base_fraction':.7,'local_fraction':.15,'v_weight':1.,'transfer_multiplier':1.,'query_block_size':64}
    control=HistoryArchive(SparseHistoryConfig(method='transfer_vaware_hybrid_history',history_density=.25,method_params=params),
                           spatial_height=height,spatial_width=width)
    old=[]
    for frame in range(1,history+1):
        k=torch.randn(1,tokens,heads,dim,device='cuda',dtype=torch.bfloat16)
        v=torch.randn_like(k)
        raw_k,raw_v=k.cpu(),v.cpu()
        archive.index_frame(0,frame,k,v,storage_k=raw_k,storage_v=raw_v)
        rotated=archive_rope0_key(k,spatial_height=height,spatial_width=width,freqs=freqs)
        control.index_frame(0,frame,rotated,raw_v,storage_k=raw_k,storage_v=raw_v)
        old.append((raw_k,raw_v))
    system=LongLiveSystemConfig(transfer_layout='exact_compact',staging_mode='persistent_separate',
        cpu_pack_policy='archive_runs',gpu_union_cache='per_chunk',gpu_union_cache_budget_mib=256,
        host_pinned_budget_mib=128,archive_offload='pooled_pageable')
    pool=PinnedStagingPool(slots=2,budget_bytes=128*1024**2,pin_memory=True)
    cache=HistoryUnionCache(256*1024**2)
    module=runtime.SparseHistorySelfAttention(dim=heads*dim,num_heads=heads,local_attn_size=local,
        sink_size=1,memory_size=history,layer_id=0,history_archive=archive,sparse_config=cfg,
        system_config=system,history_union_cache=cache,history_staging_pool=pool).cuda().bfloat16()
    module.archive_offload_stager=ArchiveOffloadStager(pool)
    module.max_attention_size=local*tokens
    base_k=torch.randn(1,local*tokens,heads,dim,device='cuda',dtype=torch.bfloat16)
    base_v=torch.randn_like(base_k)
    kv={'k':base_k.clone(),'v':base_v.clone(),'global_end_index':torch.tensor([current],device='cuda'),
        'local_end_index':torch.tensor([local*tokens],device='cuda'),
        'cpu_k_frames':[k.unsqueeze(1) for k,v in old], 'cpu_v_frames':[v.unsqueeze(1) for k,v in old]}
    captures=[]
    original_execute=runtime.execute_plan
    def capture_backend(backend,q,ek,ev,hk,hv,plan):
        result=original_execute(backend,q,ek,ev,hk,hv,plan)
        captures.append((q.clone(),ek.clone(),ev.clone(),result.output.clone(),plan))
        return result
    runtime.execute_plan=capture_backend
    try:
        x=torch.randn(1,new*tokens,heads*dim,device='cuda',dtype=torch.bfloat16)
        for call in range(5):
            output,update=module(x+call*.1,torch.tensor([new*tokens],device='cuda'),
                torch.tensor([[new,height,width]],device='cuda'),freqs,None,kv_cache=kv,
                current_start=current,memory_indices=torch.arange(history,device='cuda').view(1,-1))
            runtime._UPSTREAM.CausalWanModel._apply_cache_updates(None,[kv],[(0,update)])
            if not torch.isfinite(output).all():
                raise RuntimeError('nonfinite real module output')
    finally:
        runtime.execute_plan=original_execute
    if len(captures)!=5 or (cache.hits,cache.misses)!=(4,1):
        raise RuntimeError('real cache/forward path did not execute five calls')
    first_query=captures[0][0]
    reference_plan=control.route_indexed(0,summarize_query_for_pretransfer(first_query,64),
                                        list(range(1,history+1)),exact_k_tokens=captures[0][1].shape[1])
    dense_k,dense_v,frame_ids,token_ids=control.dense_history_tensors(0,list(range(1,history+1)))
    rotated_k=torch.cat([archive_rope0_key(k.cuda(),spatial_height=height,spatial_width=width,freqs=freqs)
                         for k,v in old],dim=1)
    records=[]
    for call,(q,ek,ev,actual,plan) in enumerate(captures):
        match=compare_coordinates(reference_plan,plan,token_base=tokens)
        if match['jaccard']!=1. or plan.metadata['query_summary_space']!='post_rope':
            raise RuntimeError('online phase route differs from isolated phase reference')
        teacher=routed_history_attention(q,rotated_k,dense_v.cuda(),frame_ids,token_ids,reference_plan,
                                         exact_key=ek,exact_value=ev)
        errors=output_error_metrics(teacher,actual)
        passed=errors['max_abs']<=.02 and errors['relative_l2']<=.01 and errors['one_minus_cosine']<=.001
        records.append({'call':call,'same_logical_selection':True,'bf16_vs_fp32':errors,'pass':passed})
    for frame in archive.frame_ids(0):
        actual=archive._layers[0][frame]
        expected_k,expected_v=(old[frame-1] if frame<=history else
            (base_k[:,(frame-history)*tokens:(frame-history+1)*tokens].cpu(),
             base_v[:,(frame-history)*tokens:(frame-history+1)*tokens].cpu()))
        if not torch.equal(actual.key,expected_k) or not torch.equal(actual.value,expected_v):
            raise RuntimeError('aligned indexing modified original archived KV')
        rotated=archive_rope0_key(actual.key.cuda(),spatial_height=height,spatial_width=width,freqs=freqs)
        view=rotated.permute(0,2,1,3)
        expected_prototypes=torch.stack([view[:,:,start:start+64].float().mean(2)
                                         for start in range(0,tokens,64)],dim=2).cpu()
        if not torch.equal(actual.block_centroids,expected_prototypes):
            raise RuntimeError('committed prototype mismatch, including newly evicted frames')
    result={'status':'pass' if all(r['pass'] for r in records) else 'fail',
        'scope':'real_forward_and_forced_eviction_against_isolated_same_route_fp32_reference',
        'gpu':torch.cuda.get_device_name(),'large':large,'records':records,'cache':cache.as_dict(),
        'archive_storage':archive.storage_summary(),'original_archived_KV_exact':True,
        'newly_evicted_aligned_prototypes_exact':True,'speed_claim':False}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--large',action='store_true')
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    result=run(args.large)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']!='pass':
        raise SystemExit(1)


if __name__=='__main__':
    main()
