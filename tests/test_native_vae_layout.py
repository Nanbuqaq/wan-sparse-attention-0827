from types import SimpleNamespace

import pytest
import torch

from adapters.longlive_sparse.native_vae_layout import NativeVAEMemoryFormat
from scripts.gate_native_vae_layout import tensor_bytes


def model():
    conv2=torch.nn.Conv3d(2,2,1)
    decoder=torch.nn.Sequential(torch.nn.Conv3d(2,2,3,padding=1),torch.nn.SiLU())
    return SimpleNamespace(conv2=conv2,decoder=decoder)


@pytest.mark.parametrize('mode',NativeVAEMemoryFormat.MODES)
def test_layout_preserves_parameter_values_and_restores_original_format(mode):
    m=model();weights=[p.detach().clone() for root in (m.conv2,m.decoder) for p in root.parameters()]
    x=torch.randn(1,2,3,5,5)
    with torch.no_grad():baseline=m.decoder(m.conv2(x))
    with NativeVAEMemoryFormat(m,mode) as ctx,torch.no_grad():
        actual=m.decoder(m.conv2(x));stats=ctx.record()
    assert torch.allclose(actual,baseline,atol=1e-6,rtol=1e-5)
    assert all(torch.equal(p,b) for p,b in zip([p for root in (m.conv2,m.decoder) for p in root.parameters()],weights))
    assert m.conv2.weight.is_contiguous() and m.decoder[0].weight.is_contiguous()
    assert not m.conv2._forward_pre_hooks and not m.decoder[0]._forward_pre_hooks
    assert (stats['input_hook_calls']>0)==(mode in ('entry_layout','all_conv_inputs'))


def test_hook_handles_cache_tensor_and_exception_cleanup():
    class Cached(torch.nn.Conv3d):
        def forward(self,x,cache_x=None):
            assert x.is_contiguous(memory_format=torch.channels_last_3d)
            assert cache_x.is_contiguous(memory_format=torch.channels_last_3d)
            raise RuntimeError('consumer fails')
    m=model();m.conv2=Cached(2,2,1)
    x=torch.randn(1,2,3,4,5)
    with pytest.raises(RuntimeError,match='consumer fails'):
        with NativeVAEMemoryFormat(m,'entry_layout'):m.conv2(x,cache_x=x)
    assert not m.conv2._forward_pre_hooks and m.conv2.weight.is_contiguous()


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError):NativeVAEMemoryFormat(model(),'pretend_fast')


def test_2d_convolutions_use_channels_last_not_the_3d_format():
    m=model();m.decoder=torch.nn.Conv2d(2,3,3,padding=1)
    x=torch.randn(2,2,5,7)
    with torch.no_grad():baseline=m.decoder(x)
    with NativeVAEMemoryFormat(m,'all_conv_inputs') as ctx,torch.no_grad():
        actual=m.decoder(x);assert m.decoder.weight.is_contiguous(memory_format=torch.channels_last)
    assert torch.allclose(actual,baseline,atol=1e-6,rtol=1e-5)
    assert m.decoder.weight.is_contiguous() and ctx.input_reformats==1


def test_fingerprint_is_independent_of_degenerate_channels_last_strides():
    base=torch.arange(12).reshape(3,4,1,1,1).bfloat16()
    changed=base.contiguous(memory_format=torch.channels_last_3d)
    assert tensor_bytes(changed)==tensor_bytes(base)
    scalar=torch.tensor([1.5],dtype=torch.bfloat16)
    unusual=scalar.as_strided((1,),(48,))
    assert tensor_bytes(unusual)==tensor_bytes(scalar)
    assert tensor_bytes(torch.tensor(1.,dtype=torch.float32))==tensor_bytes(torch.tensor([1.]))
