import pytest
import torch

from adapters.longlive_sparse.episodic_snapshot_probe import EpisodicSnapshotProbe


def caches():
    return [dict(k=torch.arange(48.).reshape(1,24,2,1),v=torch.arange(48.,96.).reshape(1,24,2,1),
                 local_end_index=torch.tensor(24),global_end_index=torch.tensor(96)) for _ in range(2)]


def test_bounded_snapshot_copies_and_restores_only_surviving_slots():
    source=caches(); probe=EpisodicSnapshotProbe(frames=6,frame_tokens=2,next_chunk_frames=3)
    event=probe.capture(source,completed_frames=48,version='pre_recache')
    assert event['global_frames']==list(range(42,48))
    assert event['raw_KV_bytes']==384
    preserved=probe.layers[0].key.clone(); source[0]['k'].zero_()
    assert torch.equal(probe.layers[0].key,preserved)
    target=caches(); before=[{k:v.clone() for k,v in c.items()} for c in target]
    probe.restore(target,current_start=78)
    for saved,c,old in zip(probe.layers,target,before):
        assert torch.equal(c['k'][:,:8],old['k'][:,:8])  # sink + upcoming eviction
        assert torch.equal(c['k'][:,8:20],saved.key)
        assert torch.equal(c['v'][:,8:20],saved.value)
        assert torch.equal(c['k'][:,20:],old['k'][:,20:])  # most recent two frames
        assert torch.equal(c['global_end_index'],old['global_end_index'])
        assert torch.equal(c['local_end_index'],old['local_end_index'])
    assert torch.equal(probe.layers[0].key,preserved)
    with pytest.raises(ValueError): probe.restore(target,current_start=81)


def test_budget_and_causal_guards_precede_mutation():
    with pytest.raises(ValueError,match='budget'):
        EpisodicSnapshotProbe(frames=6,frame_tokens=2,budget_bytes=383).capture(caches(),completed_frames=48,version='pre')
    probe=EpisodicSnapshotProbe(frames=6,frame_tokens=2)
    with pytest.raises(ValueError,match='uncommitted'):
        probe.capture(caches(),completed_frames=9,version='pre')
    probe.capture(caches(),completed_frames=48,version='pre')
    with pytest.raises(ValueError,match='strictly later'): probe.restore(caches(),current_start=48)
