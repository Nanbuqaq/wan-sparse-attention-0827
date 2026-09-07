import torch
from adapters.longlive_sparse.prototype_tail import build_prototype_tail, execute_weighted_tail_sdpa
from scripts.probe_prototype_tail import tail_output


def test_vectorized_tail_matches_reference_without_double_counting():
    gen = torch.Generator().manual_seed(145)
    q, k, v, ek, ev = [torch.randn(1, n, 2, 8, generator=gen) for n in (7, 14, 14, 3, 3)]
    indices = torch.tensor([[[0, 4, 8], [3, 7, 12]]])
    tail = build_prototype_tail(k, v, indices, frame_tokens=7, block_tokens=4, prototype_dtype=torch.float32)
    assert torch.equal(tail.counts.sum(-1), torch.full((1, 2), 11.))
    def selected(tensor):
        return tensor.permute(0, 2, 1, 3).gather(2, indices[..., None].expand(-1, -1, -1, 8)).permute(0, 2, 1, 3)
    output = execute_weighted_tail_sdpa(q, ek, ev, selected(k), selected(v), tail)
    capture = dict(query=q, key=k, value=v, exact_key=ek, exact_value=ev,
        frame_ids=torch.tensor([1]*7+[2]*7).view(1, 1, -1).expand(1, 2, -1),
        token_ids=torch.tensor(list(range(7))*2).view(1, 1, -1).expand(1, 2, -1))
    expected, _ = tail_output(capture, [[indices[0, 0], indices[0, 1]]], block_tokens=4, device='cpu')
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-6)
