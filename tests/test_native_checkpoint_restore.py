import pytest
import torch

from scripts.benchmark_native_restore import compact_committed_checkpoint,restore_committed_checkpoint


def cache(end=3):
    return dict(k=torch.arange(16.).reshape(1,8,1,2),v=torch.arange(16.).reshape(1,8,1,2)+1,
                local_end_index=torch.tensor([end]),global_end_index=torch.tensor([end]),
                pinned_start=torch.tensor([1]),pinned_len=torch.tensor([1]))


def test_checkpoint_owns_only_committed_slots_and_restores_metadata():
    original=cache();bank=compact_committed_checkpoint([original]);original['k'].fill_(-10)
    target=cache(0);target['k'].fill_(-5)
    restore_committed_checkpoint(bank,[target])
    assert torch.equal(target['k'][:,:3],torch.arange(6.).reshape(1,3,1,2))
    assert torch.all(target['k'][:,3:]==-5)
    assert int(target['local_end_index'])==3 and int(target['pinned_start'])==1
    assert tuple(bank[0]['k'].shape)==(1,3,1,2)


def test_checkpoint_rejects_empty_prefix_and_wrong_layer_count():
    with pytest.raises(ValueError):compact_committed_checkpoint([cache(0)])
    with pytest.raises(ValueError):restore_committed_checkpoint(compact_committed_checkpoint([cache()]),[])
