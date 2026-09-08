import torch

from adapters.longlive_sparse.native_commit_replay import owned_cpu,cache_metadata,cache_samples


def test_replay_log_samples_are_owned_and_do_not_mutate_cache():
    caches=[dict(k=torch.arange(32.).reshape(1,8,1,4),v=torch.ones(1,8,1,4),
        global_end_index=torch.tensor([8]),local_end_index=torch.tensor([8]),
        pinned_start=torch.tensor([-1]),pinned_len=torch.tensor([0])) for _ in range(30)]
    before=cache_metadata(caches);samples=cache_samples(caches)
    copy=owned_cpu(caches[0]['k']);caches[0]['k'].zero_()
    assert copy.sum()>0 and samples[0]['K'].sum()>0
    assert cache_metadata(caches)==before
