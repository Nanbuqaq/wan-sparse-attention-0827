from types import SimpleNamespace

import pytest
import torch

from adapters.longlive_sparse.native_vae_stream import NativeVAEStream


class FakeVAE:
    z_dim=1
    def __init__(self):
        self.conv2=torch.nn.Conv3d(1,1,1,bias=False)
        self.conv2.weight.data.fill_(1)
        self.calls=[];self.clear_cache()
    def clear_cache(self):self._feat_map=[None];self._conv_idx=[0]
    def decoder(self,x,feat_cache,feat_idx,first_chunk=False):
        self.calls.append(first_chunk)
        old=0 if feat_cache[0] is None else feat_cache[0]
        value=x+old;feat_cache[0]=x.clone();feat_idx[0]+=1
        return value.repeat(1,1,1 if first_chunk else 4,1,1)


def unpack(x,patch_size):return x


def test_chunk_boundaries_preserve_cache_and_first_frame_semantics():
    z=torch.arange(1,7,dtype=torch.float32).reshape(1,1,6,1,1)
    whole=NativeVAEStream(FakeVAE(),[0.,1.],unpack).decode_chunk(z)
    model=FakeVAE();stream=NativeVAEStream(model,[0.,1.],unpack)
    split=torch.cat([stream.decode_chunk(z[:,:,:2]),stream.decode_chunk(z[:,:,2:5]),stream.decode_chunk(z[:,:,5:])],dim=2)
    assert torch.equal(whole,split) and split.shape[2]==21
    assert model.calls==[True,False,False,False,False,False]
    stream.finish()
    with pytest.raises(RuntimeError):stream.decode_chunk(z[:,:,:1])
    stream.reset();assert stream.decode_chunk(z[:,:,:1]).shape[2]==1


def test_abandoned_iterator_invalidates_the_state_instead_of_silently_skipping():
    stream=NativeVAEStream(FakeVAE(),[0.,1.],unpack)
    iterator=stream.iter_decode(torch.ones(1,1,3,1,1))
    with torch.enable_grad():
        next(iterator);assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    iterator.close()
    assert stream.failed
    with pytest.raises(RuntimeError):stream.decode_chunk(torch.ones(1,1,1,1,1))


def test_compile_decoder_wraps_decoder_without_changing_semantics():
    z=torch.arange(1,7,dtype=torch.float32).reshape(1,1,6,1,1)
    reference=NativeVAEStream(FakeVAE(),[0.,1.],unpack).decode_chunk(z)
    model=FakeVAE();stream=NativeVAEStream(model,[0.,1.],unpack,compile_decoder=True)
    assert stream.decoder is not model.decoder
    compiled=stream.decode_chunk(z)
    assert torch.equal(reference,compiled) and model.calls==[True,False,False,False,False,False]


def test_geometry_and_nonpointwise_conv_are_rejected():
    stream=NativeVAEStream(FakeVAE(),[0.,1.],unpack)
    stream.decode_chunk(torch.ones(1,1,1,1,1))
    with pytest.raises(ValueError):stream.decode_chunk(torch.ones(1,1,1,2,1))
    with pytest.raises(ValueError):NativeVAEStream(SimpleNamespace(conv2=SimpleNamespace(kernel_size=(3,1,1))),[0.,1.],unpack)
