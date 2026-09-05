from __future__ import annotations

import pytest
import torch

from adapters.longlive_sparse.novelty import (
    causal_prototype_novelty,
    combine_value_and_novelty,
)


def test_novelty_uses_only_strictly_earlier_frames() -> None:
    prototypes = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]]]
    )
    frames = torch.tensor([1, 1, 2, 3])
    novelty = causal_prototype_novelty(prototypes, frames)
    assert novelty[0, 0, 0] == 1.0
    assert novelty[0, 0, 1] == 1.0
    assert novelty[0, 0, 2] == pytest.approx(0.0)
    assert 0.0 < novelty[0, 0, 3] < 1.0


def test_value_novelty_combination_is_scale_normalized() -> None:
    value = torch.tensor([[[1.0, 2.0]]])
    novelty = torch.tensor([[[2.0, 0.0]]])
    result = combine_value_and_novelty(value, novelty, novelty_weight=0.5)
    assert result.tolist() == [[[1.0, 1.0]]]


def test_novelty_rejects_future_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="one entry"):
        causal_prototype_novelty(torch.zeros(1, 1, 2, 3), torch.zeros(3))
