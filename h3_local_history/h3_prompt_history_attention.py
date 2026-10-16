"""Sampling processor: current text reads the full current document.

History queries retain prefix-only visibility; current audio/video queries read
all available keys. No persistent K/V cache is implemented by this processor.
The training processor and default sampling policy remain separate.
"""
import torch
import torch.nn.functional as F
from h3_spatial_history_flow import HistoryReadAttention,_apply_rotary_emb

class PromptHistoryReadAttention(HistoryReadAttention):
    def __init__(self,query_chunk_size=8192,query_checkpoint=False):
        super().__init__(query_chunk_size,query_checkpoint)
        self.text_policy='text_full'
        self.text_rows=None

    def __call__(self,attn,x,rotary_emb=None,attention_mask=None,is_causal=False):
        if self.text_policy=='prefix_only':return super().__call__(attn,x,rotary_emb,attention_mask,is_causal)
        assert attention_mask is None and not is_causal
        from torch.nn.attention import SDPBackend,sdpa_kernel
        p=self.prefix_rows;t=self.text_rows;n=x.shape[1];assert 0<t<=p<n
        k=attn.norm_k(attn.to_k(x).unflatten(-1,(attn.heads,-1)))
        if rotary_emb is not None:k=_apply_rotary_emb(k,*rotary_emb)
        k=k.transpose(1,2);v=attn.to_v(x).unflatten(-1,(attn.heads,-1)).transpose(1,2)
        kt,vt=k,v
        if self.text_policy=='text_current':
            kt=torch.cat((k[:,:,:t],k[:,:,p:]),dim=2);vt=torch.cat((v[:,:,:t],v[:,:,p:]),dim=2)
        result=[]
        for lo,hi,keys,values in [(0,t,kt,vt),(t,p,k[:,:,:p],v[:,:,:p]),(p,n,k,v)]:
            for start in range(lo,hi,self.query_chunk_size):
                end=min(hi,start+self.query_chunk_size)
                q=attn.norm_q(attn.to_q(x[:,start:end]).unflatten(-1,(attn.heads,-1)))
                if rotary_emb is not None:q=_apply_rotary_emb(q,rotary_emb[0][start:end],rotary_emb[1][start:end])
                q=q.transpose(1,2)
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):out=F.scaled_dot_product_attention(q,keys,values,dropout_p=0.)
                out=out.transpose(1,2).flatten(2,3);result.append(attn.to_out[1](attn.to_out[0](out)))
        return torch.cat(result,dim=1)

def install_prompt_history_attention(transformer,query_chunk_size=8192):
    processors=[]
    for block in transformer.transformer_blocks:
        processor=PromptHistoryReadAttention(query_chunk_size,False)
        block.attn.set_processor(processor);processors.append(processor)
    return processors
