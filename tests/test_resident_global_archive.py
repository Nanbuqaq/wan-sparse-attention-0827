from types import SimpleNamespace
import torch
from adapters.longlive_sparse.resident_global_archive import ResidentGlobalArchive


def test_initial_descriptor_keeps_reference_budget_without_cold_payload():
    caches=[dict(k=torch.ones(1,32,2,4),v=torch.ones(1,32,2,4),local_end_index=torch.tensor(8),global_end_index=torch.tensor(8)) for _ in range(2)]
    pipe=SimpleNamespace(use_relative_rope=False,guidance_scale=1,quantize_kv=False,num_frame_per_block=8,local_attn_size=32,sink_size=8,frame_seq_length=1,kv_cache_pos=caches)
    archive=ResidentGlobalArchive(pipe,archive_budget=4096)
    archive.last_commit=dict(end=8,phase=0.,prototype=torch.ones(4));archive._archive_last_scene(8)
    bank=archive.banks[0]
    assert bank['kv'] is None and bank['descriptor'].source_end==8
    assert bank['owned_bytes']==1040 and bank['physical_owned_bytes']==16
    assert archive.ledger['archive_D2H_KV_bytes']==0 and archive.ledger['CPU_archive_peak_tensor_bytes']==16
    assert archive.audit()['resident_global_elision']['elided_D2H_KV_bytes']==1024
