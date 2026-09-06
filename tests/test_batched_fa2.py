from dataclasses import replace
import pytest
import torch

from adapters.longlive_sparse.batched_fa2 import require_rectangular_shared_union
from test_raw_composition import route_for


def test_head_batching_requires_exact_group_order_and_no_padding():
    plan = route_for(torch.ones(1, 2, 5, dtype=torch.long), torch.arange(5).view(1, 1, -1).expand(1, 2, -1))
    require_rectangular_shared_union(plan, query_shape=(1, 2, 2, 64), exact_tokens=0, union_tokens=5)
    reordered = replace(plan, group_union_indices=plan.group_union_indices.flip(-1))
    with pytest.raises(ValueError, match='KV order'):
        require_rectangular_shared_union(reordered, query_shape=(1, 2, 2, 64), exact_tokens=0, union_tokens=5)
    padded = replace(plan, union_frame_ids=torch.tensor([[[1, 1, 1, 1, -1], [1, 1, 1, 1, 1]]]))
    with pytest.raises(ValueError, match='padded'):
        require_rectangular_shared_union(padded, query_shape=(1, 2, 2, 64), exact_tokens=0, union_tokens=5)
    split = replace(plan, query_group_sizes=torch.ones(1, 2, 2, dtype=torch.long))
    with pytest.raises(ValueError, match='one query group'):
        require_rectangular_shared_union(split, query_shape=(1, 2, 2, 64), exact_tokens=0, union_tokens=5)
