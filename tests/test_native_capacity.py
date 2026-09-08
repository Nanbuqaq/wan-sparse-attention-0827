import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from adapters.longlive_sparse.native_capacity import initialize_positive_only,install_positive_only_allocator


def pipe():
    return SimpleNamespace(guidance_scale=1.,quantize_kv=False,local_attn_size=32,
        num_frame_per_block=8,frame_seq_length=2,num_transformer_blocks=2,model_name='test',
        _dit_model=SimpleNamespace(num_heads=2,dim=8),kv_cache_pos=None,kv_cache_neg=None)


def test_positive_schema_and_values_match_locked_native_allocation():
    # Execute only the upstream allocation method with a tiny geometry, without
    # importing/constructing a model or claiming a GPU numerical gate.
    source=Path(__file__).resolve().parents[1]/'third_party/LongLive2/pipeline/causal_diffusion_inference.py'
    tree=ast.parse(source.read_text())
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CausalDiffusionInferencePipeline')
    method=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='_initialize_kv_cache')
    namespace={'torch':torch,'wan_default_config':{'test':{'num_heads':2,'head_dim':4}}}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
    native=pipe();optimized=pipe()
    namespace['_initialize_kv_cache'](native,1,torch.bfloat16,'cpu')
    initialize_positive_only(optimized,1,torch.bfloat16,'cpu')
    for a,b in zip(native.kv_cache_pos,optimized.kv_cache_pos):
        assert set(a)==set(b)
        for key in a:
            if isinstance(a[key],torch.Tensor):
                assert a[key].dtype==b[key].dtype and a[key].stride()==b[key].stride() and torch.equal(a[key],b[key])
            else:assert a[key]==b[key]
    assert len(native.kv_cache_neg)==2 and optimized.kv_cache_neg==[]


@pytest.mark.parametrize('field,value',[('guidance_scale',2.),('quantize_kv',True),('kv_cache_pos',[])])
def test_rejects_live_or_required_negative_state(field,value):
    p=pipe();setattr(p,field,value)
    with pytest.raises(ValueError):install_positive_only_allocator(p)


def test_installed_allocator_is_bound_and_requires_bf16():
    p=pipe();install_positive_only_allocator(p)
    with pytest.raises(ValueError):p._initialize_kv_cache(1,torch.float32,'cpu')
    p._initialize_kv_cache(1,torch.bfloat16,'cpu')
    assert len(p.kv_cache_pos)==2 and p.kv_cache_neg==[]
