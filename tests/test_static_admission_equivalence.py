import torch
from adapters.longlive_sparse.system_utility_route import _select_scored


def scalar_reference(selected, scores, widths, budget, allowed):
    selected = selected.clone()
    admitted = 0
    while admitted < budget:
        choices = [(float(scores[i])/int(widths[i]), -i) for i in range(scores.numel())
                   if not selected[i] and widths[i] <= budget-admitted and allowed[i]]
        if not choices:
            break
        index = -max(choices)[1]
        selected[index] = True
        admitted += int(widths[index])
    return selected, admitted


def test_static_sort_is_exactly_equivalent_to_iterated_argmax():
    generator = torch.Generator().manual_seed(27)
    for _ in range(30):
        widths = torch.tensor([64, 64, 24]*7)
        scores = torch.randint(-5, 6, (21,), generator=generator).float()
        selected = torch.rand(21, generator=generator) < .2
        allowed = torch.rand(21, generator=generator) < .8
        budget = int(torch.randint(0, 512, (), generator=generator))
        old_mask, old_bytes = scalar_reference(selected, scores, widths, budget, allowed)
        new_mask, new_bytes = _select_scored(selected, scores, widths, token_budget=budget,
                                             allowed=allowed, set_cost=None)
        assert torch.equal(old_mask, new_mask)
        assert old_bytes == new_bytes
