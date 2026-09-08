#!/usr/bin/env python3
"""Real CUDA precision plugin, forced eviction, cache and complete transfer ledger."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace
import traceback
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from adapters.longlive_sparse.history_cache import HistoryUnionCache
from adapters.longlive_sparse.staging import PinnedStagingPool
from adapters.longlive_sparse.offload import ArchiveOffloadStager
from adapters.longlive_sparse.phase_prototypes import canonical_wan_frequency_table,archive_rope0_key
from adapters.longlive_sparse.precision_runtime import PrecisionRecipe,WholeBlockPrecisionRuntime
from adapters.longlive_sparse.feature_prototypes import build_feature_tail
from adapters.longlive_sparse.offline_eval import output_error_metrics
from scripts.probe_prototype_tail import weighted_attention


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--large',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise ValueError('preserve prior gate')
    os.environ.update(LONGLIVE_CAPTURE_QKV='0',LONGLIVE_CAPTURE_COMPLETE_ATTENTION='0')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(327)
    if not torch.cuda.is_available():raise RuntimeError('real CUDA required')
    from adapters.longlive_sparse import runtime_attention as runtime
    height,width,heads,dim,new=(30,52,12,128,3) if args.large else (8,16,2,64,1)
    local,history=12,6;tokens=height*width;start=(local+history)*tokens
    freqs=canonical_wan_frequency_table(dim).cuda()
    old=[(torch.randn(1,tokens,heads,dim,device='cuda',dtype=torch.bfloat16),
          torch.randn(1,tokens,heads,dim,device='cuda',dtype=torch.bfloat16)) for _ in range(history)]
    base_k=torch.randn(1,local*tokens,heads,dim,device='cuda',dtype=torch.bfloat16);base_v=torch.randn_like(base_k)
    x=torch.randn(1,new*tokens,heads*dim,device='cuda',dtype=torch.bfloat16)
    candidate_ids=[6,3,1,5,2,4]
    report=dict(status='running',gpu=torch.cuda.get_device_name(),large=args.large,results=[],timing_claim=False,
        source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__),*sorted((ROOT/'adapters/longlive_sparse').glob('*.py'))]})
    try:
        for admission in ('mass_key_variance','mass_value','random'):
            recipe=PrecisionRecipe(admission=admission)
            cfg=SparseHistoryConfig(method='whole_block_precision_history',backend='resident_grouped_fa2',history_density=.14,
                refresh_policy='per_chunk',record_per_call=True,method_params={'precision_admission':admission})
            archive=HistoryArchive(cfg,spatial_height=height,spatial_width=width)
            system=LongLiveSystemConfig(transfer_layout='exact_compact',staging_mode='persistent_separate',cpu_pack_policy='archive_runs',
                gpu_union_cache='per_chunk',gpu_union_cache_budget_mib=256,archive_offload='pooled_pageable',
                route_metadata_mode='validated_reuse',execution_dataflow='qout_resident_grouped_fa2')
            pool=PinnedStagingPool(slots=2,budget_bytes=128*1024**2,pin_memory=True);cache=HistoryUnionCache(256*1024**2)
            module=runtime.SparseHistorySelfAttention(dim=heads*dim,num_heads=heads,local_attn_size=local,sink_size=1,
                memory_size=history,layer_id=0,history_archive=archive,sparse_config=cfg,system_config=system,
                history_union_cache=cache,history_staging_pool=pool).cuda().bfloat16()
            module.archive_offload_stager=ArchiveOffloadStager(pool);module.max_attention_size=local*tokens
            pipeline=SimpleNamespace(sparse_history_archive=archive,sparse_history_modules=[module])
            kv=dict(k=base_k.clone(),v=base_v.clone(),global_end_index=torch.tensor([start],device='cuda'),
                local_end_index=torch.tensor([local*tokens],device='cuda'),
                cpu_k_frames=[k.cpu().unsqueeze(1) for k,v in old],cpu_v_frames=[v.cpu().unsqueeze(1) for k,v in old])
            errors=[]
            with WholeBlockPrecisionRuntime(pipeline,recipe) as wrapper:
                wrapper.active=(module,{'freqs':freqs})
                for frame,(k,v) in enumerate(old,1):archive.index_frame(0,frame,k,v)
                def forbidden(*args,**kwargs):raise AssertionError('online full candidate access forbidden')
                archive.dense_history_tensors=forbidden
                original_execute=wrapper.execute
                def audited(owner,q,ek,ev,hk,hv,plan,**kwargs):
                    actual=original_execute(owner,q,ek,ev,hk,hv,plan,**kwargs)
                    # Teacher inputs are reconstructed only after the actual call.
                    full_k=torch.cat([archive_rope0_key(archive._layers[0][f].key.cuda(),spatial_height=height,
                        spatial_width=width,freqs=freqs) for f in candidate_ids],1)
                    full_v=torch.cat([archive._layers[0][f].value.cuda() for f in candidate_ids],1)
                    expected=torch.empty_like(q,dtype=torch.float32)
                    for b in range(q.shape[0]):
                        for h in range(heads):
                            valid=int(plan.group_history_counts[b,h,0])
                            ids=plan.union_frame_ids[b,h,:valid].tolist();tids=plan.union_token_ids[b,h,:valid].tolist()
                            indices=torch.tensor([candidate_ids.index(f)*tokens+t for f,t in zip(ids,tids)],device='cuda')
                            tail,_=build_feature_tail(full_k[b:b+1,:,h:h+1],full_v[b:b+1,:,h:h+1],indices[None,None],
                                frame_tokens=tokens,block_tokens=64,groups=4)
                            keys=torch.cat((ek[b,:,h],full_k[b,indices,h],tail.key[0,:,0]))
                            values=torch.cat((ev[b,:,h],full_v[b,indices,h],tail.value[0,:,0]))
                            counts=torch.cat((torch.ones(ek.shape[1]+valid,device='cuda'),tail.counts[0,0]))
                            expected[b,:,h]=weighted_attention(q[b,:,h],keys,values,counts)
                    error=output_error_metrics(expected,actual.output);errors.append(error)
                    assert error['max_abs']<=.02 and error['relative_l2']<=.01 and error['one_minus_cosine']<=.001,error
                    return actual
                wrapper.execute=audited
                for call in range(5):
                    out,update=module(x+call*.05,torch.tensor([new*tokens],device='cuda'),torch.tensor([[new,height,width]],device='cuda'),
                        freqs,None,kv_cache=kv,current_start=start,memory_indices=torch.tensor([[f-1 for f in candidate_ids]],device='cuda'))
                    runtime._UPSTREAM.CausalWanModel._apply_cache_updates(None,[kv],[(0,update)])
                    assert torch.isfinite(out).all()
                audit=wrapper.audit();stats=archive.stats.as_dict()
                assert len(wrapper.frames)==history+new
                assert sum(r['prototype_cache_hit'] for r in audit['records'])==4
                assert (cache.hits,cache.misses)==(4,1)
                bytes_total=stats['transferred_bytes']+stats['restore_index_h2d_bytes']+stats['rope_metadata_h2d_bytes']+audit['extra_metadata_H2D_bytes']
                fraction=bytes_total/audit['ledger']['candidate_KV_bytes_at_route_miss']
                assert fraction<=.25, ('complete_H2D_budget',fraction)
            for i in range(new):
                frame=archive._layers[0][history+i+1]
                assert torch.equal(frame.key,base_k[:,(i+1)*tokens:(i+2)*tokens].cpu())
                assert torch.equal(frame.value,base_v[:,(i+1)*tokens:(i+2)*tokens].cpu())
            result=dict(admission=admission,status='pass',errors=errors,audit=audit,
                raw_KV_H2D_bytes=stats['transferred_bytes'],restore_metadata_H2D_bytes=stats['restore_index_h2d_bytes'],
                rope_metadata_H2D_bytes=stats['rope_metadata_h2d_bytes'],all_history_H2D_bytes=bytes_total,
                all_history_H2D_over_first_route_candidates=fraction,original_evicted_KV_exact=True,full_candidate_API_poisoned=True)
            report['results'].append(result)
            print(json.dumps(dict(admission=admission,status='pass',full_H2D_fraction=fraction,
                worst_relative_l2=max(e['relative_l2'] for e in errors))),flush=True)
        report['status']='pass'
    except BaseException:
        report.update(status='fail',traceback=traceback.format_exc());raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')


if __name__=='__main__':main()
