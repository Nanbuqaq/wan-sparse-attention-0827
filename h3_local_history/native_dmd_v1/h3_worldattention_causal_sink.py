"""WorldAttention Stage1 sink+recent visibility for native H3.

The released Wan cache reserves sink_size inside the local attention budget.
This module leaves the historical no-sink-read H3 implementations unchanged.
"""
import torch
from h3_worldattention_causal import PackedCausalLayout,H3CausalContext

class WorldAttentionCausalLayout(PackedCausalLayout):
    def visible_indices(self,block):
        text=self.block_ids==-1
        if block==-1:return torch.where(text)[0]
        current=self.block_ids==block
        frames=self.video_frames[current&(self.video_frames>=0)]
        if not frames.numel():raise ValueError('missing current video')
        end=int(frames.max())+1;budget=self.local_frames-self.sink_frames
        if budget<1 or int(frames.max()-frames.min()+1)>budget:raise ValueError('current chunk exceeds sink-reserved window')
        past=(self.block_ids>=0)&(self.block_ids<block)&(self.video_frames>=0)
        keep=(self.video_frames<self.sink_frames)|(self.video_frames>=max(self.sink_frames,end-budget))
        return torch.cat((torch.where(text)[0],torch.where(past&keep)[0],torch.where(current)[0]))

class H3WorldAttentionCausalAttention:
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
        if not isinstance(layout,WorldAttentionCausalLayout):raise ValueError('explicit sink-inclusive WorldAttention layout required')
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
                budget=layout.local_frames-layout.sink_frames
                keep=(past.frames<layout.sink_frames)|(past.frames>=max(layout.sink_frames,end-budget))
                text=layout.block_ids[visible]==-1
                keys=torch.cat((keys[:,:,text],past.key[:,:,keep],keys[:,:,~text]),dim=2)
                values=torch.cat((values[:,:,text],past.value[:,:,keep],values[:,:,~text]),dim=2)
            attended = flash_attention(q.index_select(2, query_rows), keys, values, self.backend)
            output = output.index_copy(2, query_rows, attended)
        if context.write_bank is not None:
            frames = layout.video_frames
            mask = frames >= 0
            context.write_bank.commit(self.layer_id, k[:, :, mask], v[:, :, mask], frames[mask])
        return attn.to_out[1](attn.to_out[0](output.transpose(1, 2).flatten(2, 3)))


def install_worldattention_causal_attention(transformer):
    processors=[]
    for layer,block in enumerate(transformer.transformer_blocks):
        processor=H3WorldAttentionCausalAttention(layer);block.attn.set_processor(processor);processors.append(processor)
    return processors
