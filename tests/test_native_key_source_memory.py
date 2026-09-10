import pytest
import torch
from adapters.longlive_sparse.native_key_source_memory import pack_uniform_partition
from adapters.longlive_sparse.native_key_source_memory import NativeKeySourceMemory
from adapters.longlive_sparse.native_causal_block_memory import CausalBlockConfig
from types import SimpleNamespace


@pytest.mark.parametrize('tokens,width',[(7040,55),(1024,64)])
def test_compact_partition_preserves_every_coordinate_and_storage(tokens,width):
    groups=[list(range(i,i+width)) for i in range(0,tokens,width)]
    indices,counts=pack_uniform_partition(groups,tokens)
    assert indices.dtype==torch.int32 and indices.tolist()==groups
    assert int(counts.sum())==tokens and set(counts.tolist())=={width}
    assert indices.numel()*indices.element_size()==tokens*4


def test_missing_duplicate_or_unqualified_width_is_rejected():
    for groups,n in [([[0,1],[1,3]],4),([[0,1],[2]],3),([list(range(65))],65)]:
        with pytest.raises(ValueError):pack_uniform_partition(groups,n)


def test_key_source_constructor_preserves_grid_and_original_method_identity():
    model=torch.nn.Linear(1,1)
    model.blocks=[SimpleNamespace(self_attn=SimpleNamespace(_research_inplace_cache=True))]
    model.t_scale=1;model.rope_method='linear';model.original_seq_len=None
    pipe=SimpleNamespace(use_relative_rope=False,guidance_scale=1,quantize_kv=False,
        num_frame_per_block=8,sampling_steps=4,generator=SimpleNamespace(_compiled_model_call=None),
        frame_seq_length=128,_dit_model=model,local_attn_size=32,global_sink_size=8,sink_size=8,kv_cache_pos=None)
    config=CausalBlockConfig(policy='mass_value',fraction=.25,grouping='key_bank',head_policy='per_head')
    runtime=NativeKeySourceMemory(pipe,config,(8,16))
    assert runtime.grid==(8,16) and runtime.config is config and runtime.config.grouping=='key_bank'
