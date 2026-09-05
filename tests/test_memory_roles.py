from __future__ import annotations

import torch

from adapters.longlive_sparse.memory_roles import (
    build_three_role_probabilities,
    causal_query_identity_probability,
    causal_state_change_score,
)


def test_state_change_uses_latest_matching_past_spatial_block() -> None:
    values = torch.tensor(
        [[[[0.0], [10.0], [2.0], [10.0], [5.0], [13.0]]]]
    )
    frames = torch.tensor([0, 0, 1, 1, 2, 2])
    starts = torch.tensor([0, 64, 0, 64, 0, 64])
    score = causal_state_change_score(values, frames, starts)
    assert score[0, 0, 0] == 0
    assert score[0, 0, 1] == 0
    assert score[0, 0, 4] > score[0, 0, 2]


def test_three_roles_are_normalized_and_identity_has_priority() -> None:
    identity = torch.tensor([[[0.8, 0.0]]])
    state = torch.tensor([[[1.0, 0.6]]])
    roles = build_three_role_probabilities(identity, state)
    assert roles.shape == (1, 1, 2, 3)
    torch.testing.assert_close(roles.sum(dim=-1), torch.ones(1, 1, 2))
    assert roles[0, 0, 0, 0] == 0.8
    assert roles[0, 0, 0, 2] < 0.21


def test_query_role_uses_current_q_past_prototypes_and_optional_prior() -> None:
    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    keys = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    identity = torch.tensor([[[1.0, 0.0]]])
    probability = causal_query_identity_probability(query, keys, identity)
    assert probability[0, 0, 0] > probability[0, 0, 1]
    with_prior = causal_query_identity_probability(
        query, keys, identity, previous_spatial_prior=torch.ones(1, 1, 2)
    )
    assert torch.all(with_prior >= probability)
