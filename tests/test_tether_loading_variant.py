import torch

from scripts.run_tether_loading_variant import defer_constructor_cuda, tensor_return_adapter


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


def test_fa3_container_adapter_does_not_change_or_copy_the_output():
    output = torch.randn(2, 3, 4)
    assert tensor_return_adapter(lambda: (output, torch.zeros(1)))() is output
    assert tensor_return_adapter(lambda: output)() is output
