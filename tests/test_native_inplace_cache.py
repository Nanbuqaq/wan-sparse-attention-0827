import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from adapters.longlive_sparse.native_inplace_cache import transform_forward,metadata_only_updates


def upstream_functions():
    source=(Path(__file__).resolve().parents[1]/'third_party/LongLive2/wan_5b/modules/causal_model.py').read_text()
    tree=ast.parse(source)
    values={}
    for cls in tree.body:
        if isinstance(cls,ast.ClassDef):
            for fn in cls.body:
                if isinstance(fn,ast.FunctionDef) and (cls.name,fn.name) in (
                    ('CausalWanSelfAttention','forward'),('CausalWanModel','_apply_cache_updates')):
                    values[fn.name]=ast.unparse(fn)
    return values


@torch.no_grad()
def test_actual_upstream_control_flow_matches_after_inserts_rolls_pins_and_repeated_steps():
    functions=upstream_functions()
    def attention(q,k,v):
        logits=torch.einsum('bqhd,bkhd->bhqk',q,k)/(q.shape[-1]**.5)
        return torch.einsum('bhqk,bkhd->bqhd',logits.softmax(-1),v)
    scope=dict(torch=torch,math=math,_CURRENT_GRID_META={},_CGRAPH_OUTPLACE_KV_ENABLED=False,
               causal_rope_apply=lambda x,*a,**kw:x,attention=attention)
    exec(functions['forward'],scope);original=scope['forward']
    optimized_scope=dict(scope);exec(compile(transform_forward(functions['forward']),'<test>','exec'),optimized_scope)
    optimized=optimized_scope['forward'];exec(functions['_apply_cache_updates'],scope);apply=scope['_apply_cache_updates']
    def owner(enabled):
        return SimpleNamespace(num_heads=2,head_dim=4,sink_size=1,global_sink_size=1,
            local_attn_size=4,max_attention_size=8,q=torch.nn.Identity(),k=torch.nn.Identity(),
            v=torch.nn.Identity(),o=torch.nn.Identity(),norm_q=torch.nn.Identity(),norm_k=torch.nn.Identity(),
            _research_inplace_cache=enabled)
    def cache():
        return dict(k=torch.zeros(1,8,2,4),v=torch.zeros(1,8,2,4),
                    global_end_index=torch.tensor([0]),local_end_index=torch.tensor([0]),
                    pinned_start=torch.tensor([-1]),pinned_len=torch.tensor([0]))
    ca,cb=cache(),cache();rng=torch.Generator().manual_seed(615)
    for frame in range(0,12,2):
        for phase in range(2):
            x=torch.randn(1,4,8,generator=rng)
            kw=dict(seq_lens=torch.tensor([4]),grid_sizes=torch.tensor([[2,1,2]]),freqs=None,block_mask=None,current_start=frame*2)
            ya,ia=original(owner(False),x,kv_cache=ca,**kw)
            yb,ib=optimized(owner(True),x,kv_cache=cb,**kw)
            assert torch.equal(ya,yb)
            apply(None,[ca],[(0,ia)])
            updates,stats=metadata_only_updates([cb],[(0,ib)]);apply(None,[cb],updates)
            assert stats['calls']==1
            for key in ca:assert torch.equal(ca[key],cb[key]),(frame,phase,key)
        if frame in (2,6):
            for c in (ca,cb):
                c['pinned_start'].fill_(int(c['local_end_index'])-4);c['pinned_len'].fill_(2)


def test_no_grad_is_required_before_inplace_mutation():
    functions=upstream_functions();scope=dict(torch=torch)
    exec(compile(transform_forward(functions['forward']),'<test>','exec'),scope)
    import pytest
    with pytest.raises(RuntimeError,match='no-grad'):
        scope['forward'](SimpleNamespace(_research_inplace_cache=True),None,None,None,None,None)
