import pytest
import torch
from adapters.longlive_sparse.native_key_source_memory import pack_uniform_partition


@pytest.mark.parametrize('tokens,width',[(7040,55),(1024,64)])
def test_compact_partition_preserves_every_coordinate_and_storage(tokens,width):
    groups=[list(range(i,i+width)) for i in range(0,tokens,width)]
    indices,counts=pack_uniform_partition(groups,tokens)
    assert indices.dtype==torch.int32 and indices.tolist()==groups
    assert int(counts.sum())==tokens and set(counts.tolist())=={width}
    assert indices.numel()*indices.element_size()==tokens*4


def test_missing_duplicate_or_unqualified_width_is_rejected():
    for groups,n in [([[0,1],[1,3]],4),([[0,1],[2]],3),([list(range(65))],65)]:
        with pytest.raises(ValueError):pack_uniform_partition(groups,n)
