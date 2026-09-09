"""Execute the locked official block's forward with explicit test doubles.

This checks computation order, not trained-weight GPU semantics or quality.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def test_official_block_self_attention_precedes_current_text_conditioning():
    path=Path(__file__).resolve().parents[1]/'third_party/LongLive2/wan_5b/modules/causal_model.py'
    tree=ast.parse(path.read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='CausalWanAttentionBlock')
    forward=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='forward')
    module=ast.fix_missing_locations(ast.Module(body=[forward],type_ignores=[]))
    namespace=dict(torch=torch,_TRITON_ADALN_ENABLED=False,_CURRENT_GRID_META={})
    exec(compile(module,str(path),'exec'),namespace)
    calls=[]
    def self_attention(x,*args,**kwargs):
        calls.append(x.detach().clone());return torch.zeros_like(x)
    def cross_attention(x,context,*args,**kwargs):return context.expand_as(x)
    block=SimpleNamespace(modulation=torch.zeros(1,6,8),norm1=torch.nn.LayerNorm(8,elementwise_affine=False),
        norm2=torch.nn.LayerNorm(8,elementwise_affine=False),norm3=torch.nn.Identity(),
        self_attn=self_attention,cross_attn=cross_attention,ffn=torch.nn.Identity())
    x=torch.arange(64,dtype=torch.float32).reshape(1,8,8);e=torch.zeros(1,2,6,8)
    args=dict(e=e,seq_lens=torch.tensor([8]),grid_sizes=None,freqs=None,context_lens=None,block_mask=None)
    left=namespace['forward'](block,x.clone(),context=torch.zeros(1,1,8),**args)
    right=namespace['forward'](block,x.clone(),context=torch.ones(1,1,8),**args)
    assert torch.equal(calls[0],calls[1])
    assert not torch.equal(left,right)
