import torch
import pytest
from adapters.longlive_sparse.phase_prototypes import canonical_wan_frequency_table, archive_rope0_key, archive_rope0_prototypes
from adapters.longlive_sparse.rope import apply_selected_rope, build_sparse_positions


def test_index_time_prototype_equals_executed_key_mean_and_is_chunk_invariant():
    generator=torch.Generator().manual_seed(49)
    key=torch.randn(1,130,2,16,generator=generator).bfloat16()
    freqs=canonical_wan_frequency_table(16)
    proto=archive_rope0_prototypes(key,spatial_height=10,spatial_width=13,freqs=freqs)
    for current in (9,21,239):
        frames=torch.full((1,2,130),3)
        tokens=torch.arange(130).view(1,1,-1).expand(1,2,-1)
        positions=build_sparse_positions(frame_ids=frames,token_ids=tokens,current_frame_id=current,
            spatial_width=13,rope_policy='upstream_zero',max_relative_age=1023)
        executed=apply_selected_rope(key,positions,freqs)
        expected=torch.stack([executed[:,start:start+64].float().mean(1) for start in range(0,130,64)],dim=2)
        assert torch.equal(proto,expected)
    with pytest.raises(ValueError,match='only valid'):
        archive_rope0_key(key,spatial_height=10,spatial_width=13,freqs=freqs,rope_policy='recency_rank')
