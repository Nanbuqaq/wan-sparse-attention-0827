#!/usr/bin/env python3
"""Real small attention forward with causal prototype reference hooks and cache."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from adapters.longlive_sparse.history_cache import HistoryUnionCache
from adapters.longlive_sparse.staging import PinnedStagingPool
from adapters.longlive_sparse.prototype_reference_runtime import PrototypeReferenceRuntime
from adapters.longlive_sparse.phase_prototypes import canonical_wan_frequency_table


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('preserve old gate')
    torch.set_num_threads(2)
    torch.manual_seed(20260907)
    if not torch.cuda.is_available():
        raise RuntimeError('real CUDA required')
    from adapters.longlive_sparse.runtime_attention import SparseHistorySelfAttention
    from adapters.longlive_sparse import runtime_attention as runtime
    cfg = SparseHistoryConfig(method='transfer_vaware_hybrid_history', backend='resident_grouped_fa2', refresh_policy='per_chunk', history_density=.25,
        method_params={'base_fraction':.7, 'local_fraction':.15, 'v_weight':1., 'transfer_multiplier':1.})
    system = LongLiveSystemConfig(transfer_layout='exact_compact', cpu_pack_policy='archive_runs',
        staging_mode='persistent_separate', gpu_union_cache='per_chunk', gpu_union_cache_budget_mib=128,
        execution_dataflow='qout_resident_grouped_fa2', route_metadata_mode='validated_reuse')
    archive = HistoryArchive(cfg, spatial_height=8, spatial_width=16)
    for frame in (1, 2):
        archive.index_frame(0, frame, torch.randn(1,128,2,64,dtype=torch.bfloat16), torch.randn(1,128,2,64,dtype=torch.bfloat16))
    cache = HistoryUnionCache(128*1024**2)
    module = SparseHistorySelfAttention(dim=128, num_heads=2, local_attn_size=6, sink_size=1, memory_size=2,
        layer_id=0, history_archive=archive, sparse_config=cfg, system_config=system,
        history_union_cache=cache, history_staging_pool=PinnedStagingPool(slots=2,budget_bytes=128*1024**2,pin_memory=True)).cuda().bfloat16()
    module.max_attention_size = 6*128
    pipeline = SimpleNamespace(sparse_history_archive=archive, sparse_history_modules=[module])
    kv = dict(k=torch.randn(1,6*128,2,64,device='cuda',dtype=torch.bfloat16),
        v=torch.randn(1,6*128,2,64,device='cuda',dtype=torch.bfloat16),
        global_end_index=torch.tensor([5*128],device='cuda'), local_end_index=torch.tensor([3*128],device='cuda'))
    q = torch.randn(1,128,128,device='cuda',dtype=torch.bfloat16)
    freqs = canonical_wan_frequency_table(64).cuda()
    records = []
    for mode, block in (('sdpa_null',64), ('prototype_tail',64), ('prototype_tail',16)):
        module.clear_selection_cache(); cache.reset()
        with PrototypeReferenceRuntime(pipeline, mode=mode, block_tokens=block) as wrapper:
            for call in range(5):
                output, update = module(q, torch.tensor([128],device='cuda'), torch.tensor([[1,8,16]],device='cuda'),
                    freqs, None, kv_cache=kv, current_start=5*128,
                    memory_indices=torch.tensor([[0,1]],device='cuda'))
                runtime._UPSTREAM.CausalWanModel._apply_cache_updates(None,[kv],[(0,update)])
                assert bool(torch.isfinite(output).all())
        audit = wrapper.audit()
        assert audit['calls'] == 5
        if mode == 'prototype_tail':
            assert sum(r['prototype_cache_hit'] for r in audit['records']) == 4
            assert audit['extra_candidate_H2D_bytes'] == 2*256*2*64*2
        records.append(audit)
    result = dict(status='pass', gpu=torch.cuda.get_device_name(), scope='real_causal_forward_reference_hook_and_tail_cache',
                  mode_results=records, production_onload_efficiency_claim=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2); handle.write('\n')
    print(json.dumps({'status':'pass','modes':len(records)}))


if __name__ == '__main__':
    main()
