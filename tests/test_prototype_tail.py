import pytest
import torch

from scripts.probe_prototype_tail import tail_output
from adapters.longlive_sparse.offline_eval import dense_history_attention


@pytest.mark.parametrize('selected', [[], [0, 4, 7], list(range(9))])
def test_count_weighted_tail_is_exact_for_constant_keys_within_each_block(selected):
    gen = torch.Generator().manual_seed(109)
    q = torch.randn(1, 7, 2, 8, generator=gen)
    k = torch.randn(1, 9, 2, 8, generator=gen)
    k[:, 1:4] = k[:, :1]
    k[:, 6:9] = k[:, 5:6]
    v = torch.randn(k.shape, generator=gen)
    ek, ev = [torch.randn(1, 3, 2, 8, generator=gen) for _ in range(2)]
    capture = dict(query=q, key=k, value=v, exact_key=ek, exact_value=ev,
        frame_ids=torch.tensor([1]*5+[2]*4).view(1, 1, -1).expand(1, 2, -1),
        token_ids=torch.tensor(list(range(5))+list(range(4))).view(1, 1, -1).expand(1, 2, -1))
    output, accounting = tail_output(capture, [[torch.tensor(selected, dtype=torch.long)]*2], block_tokens=4, device='cpu')
    reference = dense_history_attention(q, torch.cat((ek, k), 1), torch.cat((ev, v), 1))
    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-6)
    assert accounting['raw_plus_approximate_coverage'] == 18


def test_duplicate_raw_tokens_are_not_double_counted():
    q = torch.zeros(1, 2, 1, 4)
    capture = dict(query=q, key=q, value=q, exact_key=q, exact_value=q,
                   frame_ids=torch.zeros(1, 1, 2, dtype=torch.long), token_ids=torch.arange(2).view(1, 1, -1))
    with pytest.raises(ValueError, match='unique'):
        tail_output(capture, [[torch.tensor([0, 0])]], device='cpu')
