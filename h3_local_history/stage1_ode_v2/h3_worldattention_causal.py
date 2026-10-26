"""H3-native chunk-causal attention and detached per-layer video KV writeback.

No learned history projection. Generation blocks and codec windows are separate.
An immutable per-forward context is passed through attention_mask so checkpoint
recomputation cannot accidentally use a later chunk's layout or mutated cache.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch


@dataclass(frozen=True)
class PackedCausalLayout:
    block_ids: torch.Tensor  # -1 text, otherwise absolute generation block ID
    video_frames: torch.Tensor  # absolute latent frame ID, -1 for non-video rows
    local_frames: int  # official dense stage1 window includes the CURRENT block
    sink_frames: int  # retained in cache; stage1 Flash does not force sink visibility

    def __post_init__(self):
        if self.block_ids.ndim != 1 or self.video_frames.shape != self.block_ids.shape:
            raise ValueError('Packed visibility metadata has invalid geometry')
        if self.local_frames < 1 or self.sink_frames < 0:
            raise ValueError('Invalid causal window')
        if not (self.block_ids == -1).any():
            raise ValueError('H3 requires an explicit text prefix')
        if ((self.block_ids == -1) & (self.video_frames >= 0)).any():
            raise ValueError('Text rows may not be video history')

    def groups(self):
        return [int(x) for x in torch.unique(self.block_ids, sorted=True).tolist()]

    def visible_indices(self, block):
        text = self.block_ids == -1
        if block == -1:
            return torch.where(text)[0]
        current = self.block_ids == block
        frames = self.video_frames[current & (self.video_frames >= 0)]
        if not frames.numel():
            raise ValueError('Every AV generation block requires video rows')
        start, end = int(frames.min()), int(frames.max())+1
        if end-start > self.local_frames:
            raise ValueError('Current block exceeds the complete local attention window')
        past = ((self.block_ids >= 0) & (self.block_ids < block) & (self.video_frames >= 0)
                & (self.video_frames >= end-self.local_frames))
        return torch.where(text | current | past)[0]


@dataclass(frozen=True)
class VideoKV:
    key: torch.Tensor  # BHND
    value: torch.Tensor
    frames: torch.Tensor


class VideoKVBank:
    def __init__(self, local_frames, sink_frames):
        self.local_frames, self.sink_frames = local_frames, sink_frames
        self.layers = {}

    def snapshot(self):
        # Entries are replaced, never edited in place; old forward contexts stay valid.
        return MappingProxyType(dict(self.layers))

    def commit(self, layer, key, value, frames):
        if torch.is_grad_enabled():
            raise ValueError('Clean-context KV writeback must run without gradients')
        if frames.ndim != 1 or key.shape != value.shape or key.shape[-2] != frames.numel():
            raise ValueError('Invalid layer KV geometry')
        if not frames.numel():
            raise ValueError('No video frames to commit')
        old = self.layers.get(layer)
        if old is not None:
            if int(old.frames.max()) >= int(frames.min()):
                raise ValueError('KV writeback overlaps or reverses committed frame IDs')
            key = torch.cat((old.key, key.detach()), dim=-2)
            value = torch.cat((old.value, value.detach()), dim=-2)
            frames = torch.cat((old.frames, frames))
        # Retain enough previous context for the NEXT block, plus the sink.
        retain = (frames < self.sink_frames) | (frames >= int(frames.max())+1-self.local_frames)
        self.layers[layer] = VideoKV(key.detach()[:, :, retain].contiguous(),
                                     value.detach()[:, :, retain].contiguous(),
                                     frames[retain].detach().clone())

    @property
    def bytes(self):
        return sum(x.key.numel()*x.key.element_size()+x.value.numel()*x.value.element_size()
                   for x in self.layers.values())


@dataclass(frozen=True)
class H3CausalContext:
    layout: PackedCausalLayout
    cached: Mapping[int, VideoKV] | None = None
    write_bank: VideoKVBank | None = None
    modulation_segments: tuple | None = None


class H3ChunkCausalAttention:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, layer_id, backend='torch_flash'):
        self.layer_id, self.backend = layer_id, backend
        self.query_checkpoint = False

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None, is_causal=False):
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
        from h3_attention_backend import flash_attention
        if is_causal or not isinstance(attention_mask, H3CausalContext):
            raise ValueError('Use the explicit AV chunk-causal context, not token-triangular attention')
        context = attention_mask;layout = context.layout
        if hidden_states.shape[1] != layout.block_ids.numel():
            raise ValueError('Packed current layout differs from residual rows')
        q = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1)))
        k = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1)))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1)).transpose(1, 2)
        if rotary_emb is not None:
            q = _apply_rotary_emb(q, *rotary_emb);k = _apply_rotary_emb(k, *rotary_emb)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        output = torch.zeros_like(q)
        past = context.cached.get(self.layer_id) if context.cached is not None else None
        if context.cached is not None and len([g for g in layout.groups() if g >= 0]) != 1:
            raise ValueError('Cached generation reads one declared current block per call')
        for group in layout.groups():
            query_rows = torch.where(layout.block_ids == group)[0]
            visible = layout.visible_indices(group)
            keys, values = k.index_select(2, visible), v.index_select(2, visible)
            if group >= 0 and past is not None:
                frames = layout.video_frames[(layout.block_ids == group) & (layout.video_frames >= 0)]
                start, end = int(frames.min()), int(frames.max())+1
                if (past.frames >= start).any():
                    raise ValueError('Cached video includes current or future latent frames')
                keep = past.frames >= end-layout.local_frames
                keys = torch.cat((past.key[:, :, keep], keys), dim=2)
                values = torch.cat((past.value[:, :, keep], values), dim=2)
            attended = flash_attention(q.index_select(2, query_rows), keys, values, self.backend)
            output = output.index_copy(2, query_rows, attended)
        if context.write_bank is not None:
            frames = layout.video_frames
            mask = frames >= 0
            context.write_bank.commit(self.layer_id, k[:, :, mask], v[:, :, mask], frames[mask])
        return attn.to_out[1](attn.to_out[0](output.transpose(1, 2).flatten(2, 3)))


def install_chunk_causal_attention(transformer, backend='torch_flash'):
    processors = []
    for layer, block in enumerate(transformer.transformer_blocks):
        processor = H3ChunkCausalAttention(layer, backend)
        block.attn.set_processor(processor);processors.append(processor)
    return processors
