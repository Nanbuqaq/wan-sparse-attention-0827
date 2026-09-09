from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from adapters.longlive_sparse.native_source_pin_lease import NativeSourcePinLease


def fixture():
    g=torch.Generator().manual_seed(4)
    cache=dict(k=torch.randn(1,32,1,128,generator=g),v=torch.randn(1,32,1,128,generator=g),
        pinned_start=torch.tensor(8),pinned_len=torch.tensor(8))
    pipe=SimpleNamespace(local_attn_size=32,global_sink_size=8,num_frame_per_block=8,frame_seq_length=1,
        kv_cache_pos=[cache],_dit_model=SimpleNamespace(rope_temporal_offset=24))
    entry=dict(at_latent=96,current_phase=24,installation=dict(admission_plan=dict(
        destination_token_range=[8,16],source_frames=list(range(40,48))),admission_plan_sha256='a'*64))
    memory=SimpleNamespace(last_commit=dict(end=104,phase=24),installations=[entry])
    def native(caches,n):
        for c in caches:c['pinned_start'].fill_(24);c['pinned_len'].fill_(n)
    return pipe,memory,native


def test_lease_changes_metadata_only_and_keeps_sampled_resident_values():
    pipe,memory,native=fixture();before={k:pipe.kv_cache_pos[0][k].clone() for k in ('k','v')}
    lease=NativeSourcePinLease(pipe,memory);lease.pin(native,pipe.kv_cache_pos,8)
    assert int(pipe.kv_cache_pos[0]['pinned_start'])==8
    assert all(torch.equal(before[k],pipe.kv_cache_pos[0][k]) for k in before)
    for frame in (104,112,120):lease.before(None,(),{'current_start':frame})
    assert [r['query_start_latent'] for r in lease.checks]==[104,112,120]
    assert lease.events[0]['requested_native_pin']==[24,8] and lease.events[0]['effective_source_pin']==[8,8]
    assert lease.ledger['extra_raw_KV_H2D_bytes']==0 and lease.ledger['metadata_GPU_write_bytes']==16


def test_real_sample_corruption_is_not_silently_accepted():
    pipe,memory,native=fixture();lease=NativeSourcePinLease(pipe,memory);lease.pin(native,pipe.kv_cache_pos,8)
    pipe.kv_cache_pos[0]['v'][:,10].add_(1)
    with pytest.raises(RuntimeError,match='source K/V changed'):lease.before(None,(),{'current_start':104})


def test_unrelated_future_native_pin_is_not_overridden():
    pipe,memory,native=fixture();lease=NativeSourcePinLease(pipe,memory);lease.pin(native,pipe.kv_cache_pos,8)
    pipe._dit_model.rope_temporal_offset=32;memory.last_commit=dict(end=136,phase=32)
    lease.before(None,(),{'current_start':128});assert lease.active is None
    lease.pin(native,pipe.kv_cache_pos,8);assert int(pipe.kv_cache_pos[0]['pinned_start'])==24
    assert len(lease.events)==1


@pytest.mark.parametrize('seed',[20260925,20260926])
def test_exact_source_lease_CLI_is_registered_and_read_only(tmp_path,seed):
    root=Path(__file__).resolve().parents[1];out=tmp_path/'not_created'
    args=[sys.executable,str(root/'scripts/run_longlive2_native_reference.py'),'--assets',str(tmp_path/'unused'),
        '--output',str(out),'--object-protocol-only','--cut-scenario','chest_revisit','--seed',str(seed),
        '--native-local-frames','32','--cfg1-positive-cache-only','--fixed-adaln-warps','16','--fixed-adaln-stages','1',
        '--object-state-memory-study','--causal-scene-memory','--causal-scene-position-policy','recent_virtual',
        '--object-state-text-control','past_settled_restatement','--chest-hybrid-study','--chest-source-pin-lease']
    result=subprocess.run(args,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['protocol']['source_pin_lease_registration'] is not None
    assert not out.exists()
    args.remove('--chest-hybrid-study')
    assert subprocess.run(args,capture_output=True,text=True,timeout=30).returncode!=0
