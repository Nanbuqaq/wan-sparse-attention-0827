#!/usr/bin/env python3
"""GPU commit/eviction/cache gates, with a separate full-materialization teacher.

The runtime full-candidate API is poisoned. The independent teacher directly
reads the small test archive only after actual output is produced; these gate
wall times are not eligible for speed measurements.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from adapters.longlive_sparse.history_cache import HistoryUnionCache
from adapters.longlive_sparse.staging import PinnedStagingPool
from adapters.longlive_sparse.offload import ArchiveOffloadStager
from adapters.longlive_sparse.phase_prototypes import canonical_wan_frequency_table, archive_rope0_key
from adapters.longlive_sparse.committed_moment_runtime import CommittedMomentRuntime
from adapters.longlive_sparse.prototype_tail import build_prototype_tail
from adapters.longlive_sparse.feature_prototypes import build_feature_tail
from adapters.longlive_sparse.offline_eval import output_error_metrics
from scripts.probe_prototype_tail import weighted_attention


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--large',action='store_true')
    args=p.parse_args()
    if args.output.exists(): raise ValueError('preserve prior gate')
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(712)
    if not torch.cuda.is_available(): raise RuntimeError('real CUDA required')
    from adapters.longlive_sparse import runtime_attention as runtime
    H,W,heads,dim,local,history,new = (30,52,12,128,12,6,3) if args.large else (8,16,2,64,6,2,1)
    tokens=H*W; start=(history+local)*tokens
    freqs=canonical_wan_frequency_table(dim).cuda()
    old=[(torch.randn(1,tokens,heads,dim,device='cuda',dtype=torch.bfloat16),
          torch.randn(1,tokens,heads,dim,device='cuda',dtype=torch.bfloat16)) for _ in range(history)]
    base_k=torch.randn(1,local*tokens,heads,dim,device='cuda',dtype=torch.bfloat16)
    base_v=torch.randn_like(base_k)
    x=torch.randn(1,new*tokens,heads*dim,device='cuda',dtype=torch.bfloat16)
    results=[]
    for grouping,block in (('spatial',64),('spatial',16),('key_kmeans',64)):
        cfg=SparseHistoryConfig(method='transfer_vaware_hybrid_history',backend='resident_grouped_fa2',
            history_density=.25,refresh_policy='per_chunk',record_per_call=True,
            method_params={'base_fraction':.7,'local_fraction':.15,'v_weight':1.,'transfer_multiplier':1.})
        archive=HistoryArchive(cfg,spatial_height=H,spatial_width=W)
        system=LongLiveSystemConfig(transfer_layout='exact_compact',cpu_pack_policy='archive_runs',
            staging_mode='persistent_separate',gpu_union_cache='per_chunk',gpu_union_cache_budget_mib=256,
            execution_dataflow='qout_resident_grouped_fa2',route_metadata_mode='validated_reuse',archive_offload='pooled_pageable')
        cache=HistoryUnionCache(256*1024**2)
        pool=PinnedStagingPool(slots=2,budget_bytes=128*1024**2,pin_memory=True)
        module=runtime.SparseHistorySelfAttention(dim=heads*dim,num_heads=heads,local_attn_size=local,
            sink_size=1,memory_size=history,layer_id=0,history_archive=archive,sparse_config=cfg,
            system_config=system,history_union_cache=cache,history_staging_pool=pool).cuda().bfloat16()
        module.archive_offload_stager=ArchiveOffloadStager(pool); module.max_attention_size=local*tokens
        pipeline=SimpleNamespace(sparse_history_archive=archive,sparse_history_modules=[module])
        kv=dict(k=base_k.clone(),v=base_v.clone(),global_end_index=torch.tensor([start],device='cuda'),
            local_end_index=torch.tensor([local*tokens],device='cuda'),
            cpu_k_frames=[k.cpu().unsqueeze(1) for k,v in old],cpu_v_frames=[v.cpu().unsqueeze(1) for k,v in old])
        errors=[]
        with CommittedMomentRuntime(pipeline,block_tokens=block,grouping=grouping) as wrapper:
            wrapper.active=(module,{'freqs':freqs})
            for frame,(k,v) in enumerate(old,1): archive.index_frame(0,frame,k,v)
            def forbidden(*args,**kwargs): raise AssertionError('full-candidate materialization is forbidden')
            archive.dense_history_tensors=forbidden
            actual_execute=runtime.execute_plan
            def audited(backend,q,ek,ev,hk,hv,plan,**kwargs):
                actual=actual_execute(backend,q,ek,ev,hk,hv,plan,**kwargs)
                # Independent offline teacher after actual online output.
                ks=torch.cat([archive_rope0_key(archive._layers[0][i].key.cuda(),spatial_height=H,
                    spatial_width=W,freqs=freqs) for i in range(1,history+1)],1)
                vs=torch.cat([archive._layers[0][i].value.cuda() for i in range(1,history+1)],1)
                indices=(plan.union_frame_ids-1)*tokens+plan.union_token_ids
                if grouping=='spatial': tail=build_prototype_tail(ks,vs,indices,frame_tokens=tokens,block_tokens=block)
                else: tail,_=build_feature_tail(ks,vs,indices,frame_tokens=tokens,block_tokens=block)
                expected=torch.empty_like(q,dtype=torch.float32)
                for b in range(q.shape[0]):
                    for h in range(heads):
                        k=torch.cat((ek[b,:,h],hk[b,:,h],tail.key[b,:,h]))
                        v=torch.cat((ev[b,:,h],hv[b,:,h],tail.value[b,:,h]))
                        counts=torch.cat((torch.ones(ek.shape[1]+hk.shape[1],device='cuda'),tail.counts[b,h]))
                        expected[b,:,h]=weighted_attention(q[b,:,h],k,v,counts)
                error=output_error_metrics(expected,actual.output)
                errors.append(error)
                assert error['max_abs']<=.02 and error['relative_l2']<=.01 and error['one_minus_cosine']<=.001, error
                return actual
            runtime.execute_plan=audited
            for call in range(5):
                output,update=module(x+call*.05,torch.tensor([new*tokens],device='cuda'),
                    torch.tensor([[new,H,W]],device='cuda'),freqs,None,kv_cache=kv,current_start=start,
                    memory_indices=torch.arange(history,device='cuda')[None])
                runtime._UPSTREAM.CausalWanModel._apply_cache_updates(None,[kv],[(0,update)])
                assert torch.isfinite(output).all()
            audit=wrapper.audit()
            assert audit['extra_candidate_H2D_bytes']==0
            assert len(wrapper.frames)==history+new
            assert sum(r['prototype_cache_hit'] for r in audit['records'])==4
            assert (cache.hits,cache.misses)==(4,1)
            # Detect RoPE mutation even if a cache-hit route is unchanged.
            changed=freqs.clone(); changed[0,0]*=2
            try: wrapper._frequencies(changed)
            except RuntimeError: pass
            else: raise AssertionError('changed RoPE was not rejected')
        for offset in range(new):
            frame=archive._layers[0][history+offset+1]
            assert torch.equal(frame.key,base_k[:,(offset+1)*tokens:(offset+2)*tokens].cpu())
            assert torch.equal(frame.value,base_v[:,(offset+1)*tokens:(offset+2)*tokens].cpu())
        results.append(dict(grouping=grouping,block_tokens=block,errors=errors,audit=audit,
                            original_evicted_KV_exact=True,full_candidate_API_poisoned=True))
        print(json.dumps(dict(grouping=grouping,block_tokens=block,status='pass',worst_relative_l2=max(e['relative_l2'] for e in errors))),flush=True)
    report=dict(status='pass',gpu=torch.cuda.get_device_name(),large=args.large,results=results,
        timing_eligible=False,scope='causal_commit_forced_eviction_five_pass_cache_and_independent_FP32_teacher')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(report,handle,indent=2);handle.write('\n')


if __name__=='__main__':main()
