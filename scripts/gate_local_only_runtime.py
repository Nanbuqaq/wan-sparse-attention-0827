#!/usr/bin/env python3
"""Same-backbone local-only adapter versus unchanged upstream attention."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.system_config import LongLiveSystemConfig
from adapters.longlive_sparse.phase_prototypes import canonical_wan_frequency_table


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('preserve previous gate')
    if not torch.cuda.is_available():raise RuntimeError('real CUDA required')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(620)
    from adapters.longlive_sparse import runtime_attention as runtime
    results=[]
    for height,width,heads,dim,new in ((8,16,2,64,1),(30,52,12,128,3)):
        tokens=height*width;local=12;start=20*tokens
        cfg=SparseHistoryConfig(method='native_block',backend='resident_grouped_fa2',history_density=1.,refresh_policy='per_chunk')
        archive=HistoryArchive(cfg,spatial_height=height,spatial_width=width)
        system=LongLiveSystemConfig(local_rope_layout='direct_output',execution_dataflow='qout_resident_grouped_fa2')
        options=dict(dim=heads*dim,num_heads=heads,local_attn_size=local,sink_size=1,memory_size=0)
        reference=runtime._UPSTREAM.CausalWanSelfAttention(**options).cuda().bfloat16()
        current=runtime.SparseHistorySelfAttention(**options,layer_id=0,history_archive=archive,sparse_config=cfg,system_config=system).cuda().bfloat16()
        current.load_state_dict(reference.state_dict());reference.max_attention_size=current.max_attention_size=local*tokens
        k=torch.randn(1,local*tokens,heads,dim,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
        def cache():return dict(k=k.clone(),v=v.clone(),global_end_index=torch.tensor([start],device='cuda'),local_end_index=torch.tensor([local*tokens],device='cuda'))
        caches=[cache(),cache()];x=torch.randn(1,new*tokens,heads*dim,device='cuda',dtype=torch.bfloat16)
        freqs=canonical_wan_frequency_table(dim).cuda();rows=[]
        for call in range(5):
            outputs=[]
            for model,kv in zip((reference,current),caches):
                output,update=model(x+call*.05,torch.tensor([new*tokens],device='cuda'),torch.tensor([[new,height,width]],device='cuda'),
                    freqs,None,kv_cache=kv,current_start=start,memory_indices=None)
                runtime._UPSTREAM.CausalWanModel._apply_cache_updates(None,[kv],[(0,update)])
                outputs.append(output)
            assert torch.equal(outputs[0],outputs[1]),'local-only adapter changed upstream output'
            assert torch.equal(caches[0]['k'],caches[1]['k']) and torch.equal(caches[0]['v'],caches[1]['v'])
            rows.append(dict(call=call,output_and_local_KV_bitwise_equal=True))
        stats=archive.stats.as_dict()
        assert archive.archive_bytes()==0 and stats['transferred_bytes']==0 and not archive.frame_ids(0)
        results.append(dict(shape=list(x.shape),calls=rows,archive_bytes=0,history_H2D_bytes=0,
            scope='full12-frame local cache with identical weights; no old history system'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(dict(status='pass',gpu=torch.cuda.get_device_name(),results=results,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),handle,indent=2);handle.write('\n')
    print(json.dumps(dict(status='pass',shapes=len(results),calls=10)))


if __name__=='__main__':main()
