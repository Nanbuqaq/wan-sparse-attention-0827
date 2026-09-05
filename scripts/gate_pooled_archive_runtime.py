#!/usr/bin/env python3
"""Force real cache eviction and compare legacy/paged archive storage exactly."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.history_cache import HistoryUnionCache
from adapters.longlive_sparse.offload import ArchiveOffloadStager
from adapters.longlive_sparse.phase_prototypes import canonical_wan_frequency_table
from adapters.longlive_sparse.staging import PinnedStagingPool
from adapters.longlive_sparse.system_config import LongLiveSystemConfig


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--large',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    from adapters.longlive_sparse.runtime_attention import SparseHistorySelfAttention, _UPSTREAM
    height,width,heads,dim,local,history,new=(30,52,12,128,12,6,3) if args.large else (8,16,2,64,6,2,1)
    tokens=height*width
    current_frame=history+local
    current=current_frame*tokens
    torch.manual_seed(53)
    query=torch.randn(1,new*tokens,heads*dim,device='cuda',dtype=torch.bfloat16)
    base_k=torch.randn(1,local*tokens,heads,dim,device='cuda',dtype=torch.bfloat16)
    base_v=torch.randn_like(base_k)
    old=[(torch.randn(1,tokens,heads,dim,dtype=torch.bfloat16),torch.randn(1,tokens,heads,dim,dtype=torch.bfloat16)) for _ in range(history)]
    freqs=canonical_wan_frequency_table(dim).cuda()
    records=[]
    for method in ('rag_dense','transfer_vaware_hybrid_history'):
        reference=None
        reference_archive=None
        reference_routes=None
        weights=None
        for mode in ('legacy_pinned','pooled_pageable'):
            params={'base_fraction':.7,'local_fraction':.15,'transfer_multiplier':1.,'v_weight':1.} if method!='rag_dense' else {}
            config=SparseHistoryConfig(method=method,history_density=1. if method=='rag_dense' else .25,
                refresh_policy='per_chunk',rope_policy='upstream_zero',method_params=params)
            archive=HistoryArchive(config,spatial_height=height,spatial_width=width)
            for frame,(k,v) in enumerate(old,1):
                archive.index_frame(0,frame,k,v)
            system=LongLiveSystemConfig(transfer_layout='exact_compact',staging_mode='persistent_separate',
                cpu_pack_policy='archive_runs',gpu_union_cache='per_chunk',gpu_union_cache_budget_mib=256,
                host_pinned_budget_mib=128,archive_offload=mode)
            pool=PinnedStagingPool(slots=2,budget_bytes=128*1024**2,pin_memory=True)
            cache=HistoryUnionCache(256*1024**2)
            module=SparseHistorySelfAttention(dim=heads*dim,num_heads=heads,local_attn_size=local,
                sink_size=1,memory_size=history,layer_id=0,history_archive=archive,sparse_config=config,
                system_config=system,history_union_cache=cache,history_staging_pool=pool).cuda().bfloat16()
            module.max_attention_size=local*tokens
            if mode=='pooled_pageable':
                module.archive_offload_stager=ArchiveOffloadStager(pool)
            if weights is None:
                weights={key:value.clone() for key,value in module.state_dict().items()}
            module.load_state_dict(weights)
            kv={'k':base_k.clone(),'v':base_v.clone(),
                'global_end_index':torch.tensor([current],device='cuda'),
                'local_end_index':torch.tensor([local*tokens],device='cuda'),
                'cpu_k_frames':[k.unsqueeze(1) for k,v in old],
                'cpu_v_frames':[v.unsqueeze(1) for k,v in old]}
            outputs=[]
            torch.cuda.synchronize()
            start=time.perf_counter()
            for _ in range(5):
                output,update=module(query,torch.tensor([new*tokens],device='cuda'),
                    torch.tensor([[new,height,width]],device='cuda'),freqs,None,kv_cache=kv,
                    current_start=current,memory_indices=torch.arange(history,device='cuda').view(1,-1))
                # Apply the actual upstream update path, otherwise every call
                # falsely evicts again and reindexes the same global frames.
                _UPSTREAM.CausalWanModel._apply_cache_updates(None,[kv],[(0,update)])
                outputs.append(output.clone())
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            assert archive.frame_ids(0)==list(range(1,history+new+1))
            snapshots=[]
            canonical_checks=[]
            for offset in range(new):
                frame=archive._layers[0][history+offset+1]
                expected_k=base_k[:,(offset+1)*tokens:(offset+2)*tokens].cpu()
                expected_v=base_v[:,(offset+1)*tokens:(offset+2)*tokens].cpu()
                assert torch.equal(frame.key,expected_k) and torch.equal(frame.value,expected_v)
                assert frame.key.is_pinned()==(mode=='legacy_pinned')
                if frame.block_value_centroids.numel():
                    source=frame.value.permute(0,2,1,3)
                    canonical_v=torch.stack([source[:,:,start:start+64].float().mean(2)
                                             for start in range(0,tokens,64)],dim=2)
                    canonical_checks.append({'frame':history+offset+1,
                        'value_prototype_matches_committed_value':torch.equal(canonical_v,frame.block_value_centroids),
                        'max_abs_value_prototype_error':float((canonical_v-frame.block_value_centroids).abs().max())})
                snapshots.extend([frame.key.clone(),frame.value.clone(),frame.block_centroids.clone(),frame.block_value_centroids.clone()])
            print(json.dumps({'method':method,'mode':mode,'canonical_prototype_checks':canonical_checks}),flush=True)
            assert all(item['value_prototype_matches_committed_value'] for item in canonical_checks), canonical_checks
            routes=[row['route_plan_sha256'] for row in archive.stats.call_records]
            if reference is None:
                reference,reference_archive,reference_routes=outputs,snapshots,routes
            else:
                assert all(torch.equal(a,b) for a,b in zip(reference,outputs))
                print(json.dumps({'snapshot_deltas':[
                    {'frame_offset':index//4,'field':('key','value','key_prototype','value_prototype')[index%4],
                     'max_abs':float((a.float()-b.float()).abs().max()) if a.numel() else 0.}
                    for index,(a,b) in enumerate(zip(reference_archive,snapshots)) if not torch.equal(a,b)]}),flush=True)
                assert all(torch.equal(a,b) for a,b in zip(reference_archive,snapshots))
                assert reference_routes==routes
            assert (cache.hits,cache.misses)==(4,1),cache.as_dict()
            records.append({'method':method,'archive_offload':mode,'status':'pass','forced_evictions':new,
                'outputs_and_archived_kv_and_prototypes_equal':True,'route_sha_sequence':routes,
                'canonical_prototype_checks':canonical_checks,
                'archive_storage':archive.storage_summary(),'pool':pool.as_dict(),'cache':cache.as_dict(),
                'first_use_five_call_wall_s':elapsed,'performance_claim':False})
            print(json.dumps(records[-1]),flush=True)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'status':'pass','gpu':torch.cuda.get_device_name(),
        'scope':'synthetic forced eviction with actual upstream cache updates','records':records},indent=2)+'\n')


if __name__=='__main__':
    main()
