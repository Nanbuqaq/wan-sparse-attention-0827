import torch

from adapters.longlive_sparse.native_commit_replay import NativeCleanCommitLog,owned_cpu,cache_metadata,cache_samples,release_cache_containers


def test_replay_log_samples_are_owned_and_do_not_mutate_cache():
    caches=[dict(k=torch.arange(32.).reshape(1,8,1,4),v=torch.ones(1,8,1,4),
        global_end_index=torch.tensor([8]),local_end_index=torch.tensor([8]),
        pinned_start=torch.tensor([-1]),pinned_len=torch.tensor([0])) for _ in range(30)]
    before=cache_metadata(caches);samples=cache_samples(caches)
    copy=owned_cpu(caches[0]['k']);caches[0]['k'].zero_()
    assert copy.sum()>0 and samples[0]['K'].sum()>0
    assert cache_metadata(caches)==before


def test_log_payload_excludes_teacher_KV_samples():
    log=NativeCleanCommitLog(None)
    log.records=[dict(latent=torch.zeros(1),current_start=0,samples=['teacher'],metadata={'teacher':1})]
    payload=log.payload()
    assert 'samples' not in payload['records'][0] and 'metadata' not in payload['records'][0]
    assert payload['full_KV_and_attention_outputs_not_in_log']


def test_release_clears_forward_kwarg_list_aliases():
    from types import SimpleNamespace
    names=('kv_cache_pos','kv_cache_neg','crossattn_cache_pos','crossattn_cache_neg')
    pipeline=SimpleNamespace(**{n:[{'k':torch.ones(2)}] for n in names})
    retained=[getattr(pipeline,n) for n in names]
    release_cache_containers(pipeline)
    assert all(not old for old in retained)
    assert all(getattr(pipeline,n) is None for n in names)
