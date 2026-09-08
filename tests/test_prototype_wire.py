import torch
from adapters.longlive_sparse.prototype_wire import encode_prototype_frame,choose_whole_blocks,whole_block_token_indices
from adapters.longlive_sparse.feature_prototypes import build_feature_tail


def test_wire_preserves_full_block_prototype_means():
    gen=torch.Generator().manual_seed(381)
    key=torch.randn(1,14,2,8,generator=gen,dtype=torch.bfloat16);value=torch.randn(key.shape,generator=gen,dtype=torch.bfloat16)
    wire=encode_prototype_frame(key,value,block_tokens=4,groups=2)
    tail,_=build_feature_tail(key,value,torch.empty(1,2,0,dtype=torch.long),frame_tokens=14,block_tokens=4,groups=2)
    assert torch.equal(wire.counts.float(),tail.counts)
    torch.testing.assert_close(wire.key_mean.permute(0,2,1,3),tail.key,atol=0,rtol=0)
    torch.testing.assert_close(wire.value_mean.permute(0,2,1,3),tail.value,atol=0,rtol=0)
    assert wire.key_variance.dtype==torch.bfloat16 and wire.counts.dtype==torch.int16
    assert wire.bytes==3*wire.key_mean.numel()*2+wire.counts.numel()*2


def test_whole_blocks_obey_cap_and_do_not_split_frame_tail():
    scores=torch.tensor([[[4.,3.,2.,1.],[1.,2.,3.,4.]]])
    widths=torch.tensor([64,24,64,24])
    selected,counts=choose_whole_blocks(scores,widths,token_budget=100)
    assert bool((counts<=100).all())
    for h in range(2):
        indices=whole_block_token_indices(selected[0][h],frame_tokens=88,frames=2)
        assert indices.numel()==counts[0,h]
        for block in selected[0][h]:
            assert whole_block_token_indices([block],frame_tokens=88,frames=2).numel()==widths[block]


def test_budget_below_one_block_returns_empty_without_overrun():
    blocks,counts=choose_whole_blocks(torch.ones(1,1,3),torch.full((3,),64),token_budget=32)
    assert blocks==[[[]]] and counts.item()==0


def test_uint8_variance_codec_is_explicit_and_more_compact():
    key=torch.randn(1,128,2,64,dtype=torch.bfloat16)
    bf16=encode_prototype_frame(key,key)
    quantized=encode_prototype_frame(key,key,variance_codec='u8_scaled')
    assert quantized.bytes<bf16.bytes and quantized.key_variance.dtype==torch.uint8
    assert torch.equal(quantized.key_mean,bf16.key_mean)
    assert torch.isfinite(quantized.decoded_variance()).all()
    torch.testing.assert_close(quantized.decoded_variance(),bf16.decoded_variance(),atol=.02,rtol=.02)


def test_vectorized_order_is_identical_to_scalar_reference():
    gen=torch.Generator().manual_seed(433)
    scores=torch.rand(2,3,12,generator=gen);widths=torch.tensor([64,64,24]*4)
    actual,counts=choose_whole_blocks(scores,widths,token_budget=237)
    for b in range(2):
        for h in range(3):
            left=237;chosen=[]
            for i in torch.argsort(scores[b,h]/widths,descending=True,stable=True).tolist():
                if int(widths[i])<=left:chosen.append(i);left-=int(widths[i])
            assert actual[b][h]==sorted(chosen) and counts[b,h]==237-left
