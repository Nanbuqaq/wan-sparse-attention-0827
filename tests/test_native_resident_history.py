import pytest
import torch

from adapters.longlive_sparse.native_resident_history import (
    NativeResidentConfig, updated_frame_slots, window_slot_indices,
    summarize_frame, contrast_scores, choose_whole_blocks,
)


def test_owned_coordinates_follow_native_roll_not_new_contiguous_timestamps():
    previous = [0,1,10,11,12,13,14,15]
    info = dict(action='roll_and_insert',sink_tokens=20,num_evicted_tokens=20,
                num_rolled_tokens=40,local_start_index=60,local_end_index=80)
    actual = updated_frame_slots(previous, info, 160, 10)
    assert actual == [0,1,12,13,14,15,16,17]
    assert previous == [0,1,10,11,12,13,14,15]


@pytest.mark.parametrize('sink,pin,expected', [
    (True,True,[0,1,4,5,8,9,10,11]),
    (True,False,[0,1,6,7,8,9,10,11]),
    (False,True,[4,5,6,7,8,9,10,11]),
    (False,False,[4,5,6,7,8,9,10,11]),
])
def test_actual_native_window_order_with_floating_pins(sink,pin,expected):
    result = window_slot_indices(end=120,start=40,effective_sink=20,
        pinned_start=40,pinned_len=20,prepend_sink=sink,prepend_pinned=pin,
        max_tokens=80,frame_tokens=10)
    assert result == expected


def test_frame880_has_a_real48_token_tail_and_summary_is_owned():
    k = torch.arange(880*2*4).reshape(880,2,4).float(); v = k*2
    km,vm,count = summarize_frame(k,v)
    assert count.tolist() == [64]*13+[48]
    assert torch.equal(km[-1], k[832:].mean(0))
    copy=km.clone(); k.zero_(); v.zero_()
    assert torch.equal(copy,km)


def test_whole_block_budget_is_in_actual_tokens_and_deterministic():
    selected,used,budget=choose_whole_blocks([1,1,0],[64,48,64],.5)
    assert (selected,used,budget)==([0],64,88)
    assert choose_whole_blocks([1,2],[64,48],.25)==([],0,28)


def test_proxy_is_not_the_independent_renormalized_deletion_error():
    q=torch.zeros(1,1,2); k=torch.zeros(2,1,2)
    v=torch.tensor([[[0.,0.]],[[2.,0.]]]);counts=torch.ones(2)
    proxy=contrast_scores(q,k,v,counts,'contrast_value')
    assert torch.allclose(proxy,torch.tensor([.5,.5]))
    # Independent dense operator, then physically remove the first K/V pair.
    original=(q[:,0]@k[:,0].T).softmax(-1)@v[:,0]
    removed=(q[:,0]@k[1:,0].T).softmax(-1)@v[1:,0]
    assert torch.equal((removed-original).norm(dim=-1),torch.ones(1))
    assert not torch.equal(proxy[:1],(removed-original).norm(dim=-1))


def test_contrast_proxy_ignores_a_shared_value_translation():
    rng=torch.Generator().manual_seed(72)
    q=torch.randn(9,2,4,generator=rng);k=torch.randn(5,2,4,generator=rng)
    v=torch.randn(5,2,4,generator=rng);counts=torch.tensor([64,64,48,64,48])
    before=contrast_scores(q,k,v,counts,'contrast_value')
    after=contrast_scores(q,k,v+torch.tensor([10.,-3.,7.,1.]),counts,'contrast_value')
    assert torch.allclose(before,after,atol=2e-6)


def test_unknown_modes_fail_instead_of_falling_back():
    with pytest.raises(ValueError):NativeResidentConfig(policy='dense_fallback')
    with pytest.raises(ValueError):NativeResidentConfig(fraction=1.1)
