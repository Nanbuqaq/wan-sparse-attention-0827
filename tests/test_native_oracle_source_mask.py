import pytest
import torch
from adapters.longlive_sparse.native_oracle_source_mask import fixed_mask_indices
from adapters.longlive_sparse.native_oracle_source_mask import NativeOracleSourceMaskMemory
from adapters.longlive_sparse.native_causal_block_memory import CausalBlockConfig
from types import SimpleNamespace


def test_region_is_never_trimmed_and_fill_has_exact_unique_budget():
    foreground=torch.tensor([1,4,9])
    out=fixed_mask_indices(foreground,source_tokens=16,budget=8,mode='foreground')
    assert out.numel()==out.unique().numel()==8 and set(foreground.tolist())<=set(out.tolist())
    background=fixed_mask_indices(foreground,source_tokens=16,budget=8,mode='background')
    assert background.numel()==8 and not set(background.tolist())&set(foreground.tolist())
    full=fixed_mask_indices(foreground,source_tokens=16,budget=16,mode='foreground')
    assert torch.equal(full,torch.arange(16))
    with pytest.raises(ValueError):fixed_mask_indices(foreground,source_tokens=16,budget=2,mode='foreground')


def test_invalid_or_duplicate_source_coordinates_fail():
    for ids in ([],[1,1],[16],[-1]):
        with pytest.raises(ValueError):fixed_mask_indices(ids,source_tokens=16,budget=8,mode='foreground')


def test_oracle_cannot_force_unverified_or_wrong_causal_source(tmp_path):
    model=torch.nn.Linear(1,1);model.blocks=[SimpleNamespace(self_attn=SimpleNamespace(_research_inplace_cache=True))]
    model.t_scale=1;model.rope_method='linear';model.original_seq_len=None
    pipe=SimpleNamespace(use_relative_rope=False,guidance_scale=1,quantize_kv=False,
        num_frame_per_block=8,sampling_steps=4,generator=SimpleNamespace(_compiled_model_call=None),
        frame_seq_length=128,_dit_model=model,local_attn_size=32,global_sink_size=8,sink_size=8,kv_cache_pos=None)
    path=tmp_path/'mask.pt';torch.save({'source_start':8,'source_end':16,'token_grid':[8,16],
        'indices':torch.arange(0,1024,4),'source_latent_sha256':'fixture'},path)
    runtime=NativeOracleSourceMaskMemory(pipe,CausalBlockConfig(policy='source_mask',fraction=.25),(8,16),mask_path=path)
    runtime.active_bank={'descriptor':SimpleNamespace(source_end=16)}
    with pytest.raises(RuntimeError):runtime.mask_source_indices(0,24,256)
    runtime.source_verified=True;runtime.active_bank['descriptor'].source_end=48
    with pytest.raises(RuntimeError):runtime.mask_source_indices(0,24,256)
    runtime.active_bank['descriptor'].source_end=16
    assert runtime.mask_source_indices(0,24,256).shape==(24,256)
