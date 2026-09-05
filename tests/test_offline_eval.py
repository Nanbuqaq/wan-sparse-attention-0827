from __future__ import annotations

import torch

from adapters.longlive_sparse.ar_routing import route_history
from adapters.longlive_sparse.offline_eval import (
    dense_history_attention,
    output_error_metrics,
    routed_history_attention,
)


def test_dense_route_matches_history_only_teacher() -> None:
    generator = torch.Generator().manual_seed(9)
    query = torch.randn(1, 6, 2, 4, generator=generator)
    key = torch.randn(1, 8, 2, 4, generator=generator)
    value = torch.randn(1, 8, 2, 4, generator=generator)
    frames = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).view(1, 1, 8).expand(1, 2, -1)
    tokens = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]).view(1, 1, 8).expand(1, 2, -1)
    route = route_history(
        query,
        key,
        frames,
        tokens,
        method="rag_dense",
        density=1.0,
        exact_k_tokens=0,
    )
    dense = dense_history_attention(query, key, value, query_chunk_size=2)
    routed = routed_history_attention(
        query, key, value, frames, tokens, route, query_chunk_size=2
    )
    torch.testing.assert_close(routed, dense)
    metrics = output_error_metrics(dense, routed)
    assert metrics["max_abs"] == 0.0
    assert metrics["relative_l2"] == 0.0
