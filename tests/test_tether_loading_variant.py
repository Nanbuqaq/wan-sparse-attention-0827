import torch

from scripts.run_tether_loading_variant import defer_constructor_cuda


def test_constructor_deferral_preserves_weights_and_restores_cuda_method():
    layer = torch.nn.Linear(3, 4)
    before = {k: v.clone() for k, v in layer.state_dict().items()}
    original = torch.nn.Module.cuda
    with defer_constructor_cuda() as calls:
        assert layer.cuda() is layer
        assert calls == ["Linear"]
        assert all(p.device.type == "cpu" for p in layer.parameters())
    assert torch.nn.Module.cuda is original
    assert all(torch.equal(before[k], v) for k, v in layer.state_dict().items())
