import pytest
import torch
from adapters.longlive_sparse.global_payload_control import global_read_buffers


def test_zero_values_preserve_every_key_and_do_not_mutate_cached_storage():
    k=torch.randn(1,8,2,4);v=torch.randn_like(k);old=k.clone(),v.clone()
    rk,rv,owned=global_read_buffers(k,v,2,'zero_v')
    assert rk is k and torch.equal(k,old[0]) and torch.equal(v,old[1])
    assert not rv[:,:2].count_nonzero() and torch.equal(rv[:,2:],v[:,2:])
    assert owned==v.numel()*v.element_size() and rv.data_ptr()!=v.data_ptr()


def test_zero_keys_and_values_keeps_real_token_positions_and_exact_noop_control():
    k=torch.randn(1,8,2,4);v=torch.randn_like(k)
    rk,rv,owned=global_read_buffers(k,v,2,'zero_kv')
    assert not rk[:,:2].count_nonzero() and not rv[:,:2].count_nonzero()
    assert torch.equal(rk[:,2:],k[:,2:]) and torch.equal(rv[:,2:],v[:,2:])
    assert owned==2*v.numel()*v.element_size()
    rk,rv,owned=global_read_buffers(k,v,2,'original');assert rk is k and rv is v and owned==0
    with pytest.raises(ValueError):global_read_buffers(k,v,8,'zero_v')
