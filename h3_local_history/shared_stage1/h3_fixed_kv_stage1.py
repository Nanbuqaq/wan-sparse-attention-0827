"""H3 clean/noisy TF routing, following RAVEN-H3's split/cache protocol.

RAVEN reference: 692ec211d4cd37a99a04773186fc2bf810430f78,
projects/minimax_h3/{meta_models/causal_minimax_h3_tf.py,
modeling/transformer/causal_model.py}. This local Diffusers adaptation preserves
native velocity heads and video-only source-PTS history rather than RAVEN's
x0 wrapper and AV history. It is not a RAVEN quality reproduction.
"""
from dataclasses import dataclass
from types import MappingProxyType
import types

import torch
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class Split:
    role: str  # text, clean video, or noisy joint AV
    block: int
    start: int
    end: int
    past_blocks: tuple


@dataclass(frozen=True)
class KV:
    key: torch.Tensor
    value: torch.Tensor
    block: int


class VideoBank:
    """Replace entries atomically; old snapshots remain valid for backward."""
    def __init__(self, detach=True):
        self.layers = {}
        self.detach = detach

    def snapshot(self):
        return MappingProxyType({layer:tuple(entries) for layer,entries in self.layers.items()})

    def commit(self, layer, key, value, block):
        old = self.layers.get(layer, ())
        if old and old[-1].block >= block:
            raise ValueError('Overlapping or reversed video KV commit')
        if self.detach:
            key,value = key.detach(),value.detach()
        self.layers[layer] = (*old, KV(key,value,block))

    def evict(self, keep_blocks):
        keep = set(keep_blocks)
        self.layers = {layer:tuple(x for x in entries if x.block in keep)
                       for layer,entries in self.layers.items()}

    @property
    def bytes(self):
        return sum(x.key.numel()*x.key.element_size()+x.value.numel()*x.value.element_size()
                   for entries in self.layers.values() for x in entries)


@dataclass(frozen=True)
class Context:
    splits: tuple
    text_reads_history: bool = False
    cached: object = None
    write_bank: object = None
    modulation_segments: tuple = None

    def visible_splits(self, query):
        same_text = lambda s:s.role=='text' and s.block==query.block
        past = lambda s:s.role=='clean' and s.block in query.past_blocks
        if query.role=='text':
            return tuple(s for s in self.splits if same_text(s) or (self.text_reads_history and past(s)))
        if query.role=='clean':
            return tuple(s for s in self.splits if same_text(s) or past(s)
                         or (s.role=='clean' and s.block==query.block))
        if query.role=='noisy':
            return tuple(s for s in self.splits if same_text(s) or past(s)
                         or (s.role=='noisy' and s.block==query.block))
        raise ValueError('Unknown split role')


class FixedKVAttention:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, layer):
        self.layer = layer
        self.query_chunk_size = 8192
        self.query_checkpoint = False
        self.attention_backend = 'torch_flash'

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None, is_causal=False):
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
        from h3_attention_backend import flash_attention
        if is_causal or not isinstance(attention_mask,Context):
            raise ValueError('Fixed-KV H3 requires an immutable split Context')
        context = attention_mask
        q = attn.norm_q(attn.to_q(hidden_states).unflatten(-1,(attn.heads,-1)))
        k = attn.norm_k(attn.to_k(hidden_states).unflatten(-1,(attn.heads,-1)))
        v = attn.to_v(hidden_states).unflatten(-1,(attn.heads,-1)).transpose(1,2)
        if rotary_emb is not None:
            q,k = _apply_rotary_emb(q,*rotary_emb),_apply_rotary_emb(k,*rotary_emb)
        q,k = q.transpose(1,2),k.transpose(1,2)
        saved = context.cached.get(self.layer,()) if context.cached is not None else ()
        outputs=[]
        cursor=0
        for split in context.splits:
            if split.start!=cursor or split.end<=split.start:
                raise ValueError('Splits must partition the packed document in order')
            cursor=split.end
            visible=context.visible_splits(split)
            keys=[k[:,:,s.start:s.end] for s in visible]
            values=[v[:,:,s.start:s.end] for s in visible]
            reads_past=split.role!='text' or context.text_reads_history
            if reads_past:
                for entry in saved:
                    if entry.block in split.past_blocks:
                        if entry.block>=split.block:
                            raise ValueError('Cached current/future video is forbidden')
                        keys.append(entry.key);values.append(entry.value)
            keys,values=torch.cat(keys,dim=2),torch.cat(values,dim=2)
            for lo in range(split.start,split.end,self.query_chunk_size):
                hi=min(split.end,lo+self.query_chunk_size)
                def attend(qq,kk,vv):
                    return flash_attention(qq,kk,vv,self.attention_backend)
                args=(q[:,:,lo:hi],keys,values)
                y=checkpoint(attend,*args,use_reentrant=False) if torch.is_grad_enabled() and self.query_checkpoint else attend(*args)
                outputs.append(y)
            if context.write_bank is not None and split.role=='clean':
                context.write_bank.commit(self.layer,k[:,:,split.start:split.end],v[:,:,split.start:split.end],split.block)
        if cursor!=hidden_states.shape[1]:
            raise ValueError('Unrouted rows in fixed-KV document')
        out=torch.cat(outputs,dim=2).transpose(1,2).flatten(2,3)
        return attn.to_out[1](attn.to_out[0](out))


def install(transformer):
    """Keep native parameter names; isolate prompts before joint DiT attention."""
    processors=[]
    for layer,block in enumerate(transformer.transformer_blocks):
        processor=FixedKVAttention(layer);block.attn.set_processor(processor);processors.append(processor)
    refiner=transformer.token_refiner
    if not hasattr(refiner,'_fixed_kv_original_forward'):
        refiner._fixed_kv_original_forward=refiner.forward
        def refine(self, hidden_states):
            ranges=getattr(self,'_fixed_kv_text_ranges',None)
            if ranges is None:
                raise RuntimeError('Missing independent per-block text-refiner ranges')
            return torch.cat([self._fixed_kv_original_forward(hidden_states[:,a:b]) for a,b in ranges],dim=1)
        refiner.forward=types.MethodType(refine,refiner)
    return processors


def bind(transformer,inputs,context):
    from h3_stable_modulation import segments_for
    ranges=[];offset=0
    for split in context.splits:
        if split.role=='text':
            ranges.append((offset,offset+split.end-split.start));offset+=split.end-split.start
    transformer.token_refiner._fixed_kv_text_ranges=tuple(ranges)
    return Context(context.splits,context.text_reads_history,context.cached,context.write_bank,
                   segments_for(inputs['timestep_indices']*3+inputs['token_tags']))


def forward(transformer,inputs,context):
    context=bind(transformer,inputs,context)
    return transformer(**inputs,attention_mask=context)
