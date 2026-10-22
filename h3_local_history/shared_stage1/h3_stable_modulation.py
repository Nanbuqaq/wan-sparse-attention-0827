"""Same H3 forward, FP32 segmented reduction for repeated AdaLN table indices."""
import types

import torch
from torch.utils.checkpoint import checkpoint


def segments_for(indices):
    values, counts = torch.unique_consecutive(indices, return_counts=True)
    values, counts = values.cpu().tolist(), counts.cpu().tolist()
    result=[]
    start=0
    for value,count in zip(values,counts):
        result.append((start,start+count,value))
        start+=count
    return tuple(result)


class SegmentedGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, segments):
        ctx.shape, ctx.dtype = values.shape, values.dtype
        ctx.segments=segments
        return values.index_select(0,indices)

    @staticmethod
    def backward(ctx, gradient):
        dtype=torch.float64 if ctx.dtype==torch.float64 else torch.float32
        result=torch.zeros(ctx.shape,device=gradient.device,dtype=dtype)
        for start,end,index in ctx.segments:
            result[index].add_(gradient[start:end].sum(dim=0,dtype=dtype))
        return result.to(ctx.dtype),None,None


def stable_block_forward(self,hidden_states,temb,adaln_indices,rotary_emb,attention_mask=None,is_causal=False):
    if not self._segmented_modulation_enabled:
        return self._original_modulation_forward(hidden_states,temb,adaln_indices,rotary_emb,attention_mask,is_causal)
    # New cached generation retains several forward graphs. Each checkpoint
    # must capture its own layout instead of reading a later chunk's mutable state.
    segments=getattr(attention_mask,'modulation_segments',None)
    if segments is None:
        segments=self._modulation_segments
    if segments is None:
        raise RuntimeError('Missing AdaLN segment contract for this batch')
    def gather(values):
        return SegmentedGather.apply(values,adaln_indices,segments)
    shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp=self.adaln_proj(temb)
    residual=hidden_states
    norm_hidden_states=self.norm1(hidden_states)
    norm_hidden_states=norm_hidden_states*(1.+gather(scale_msa))+gather(shift_msa)
    attention=self.attn(norm_hidden_states,rotary_emb,attention_mask,is_causal)
    hidden_states=residual+gather(gate_msa)*attention
    residual=hidden_states
    norm_hidden_states=self.norm2(hidden_states)
    norm_hidden_states=norm_hidden_states*(1.+gather(scale_mlp))+gather(shift_mlp)
    if self.ff_chunk_size and norm_hidden_states.shape[1]>self.ff_chunk_size:
        outputs=[]
        for start in range(0,norm_hidden_states.shape[1],self.ff_chunk_size):
            value=norm_hidden_states[:,start:start+self.ff_chunk_size]
            if torch.is_grad_enabled() and self.ff_chunk_checkpoint:
                outputs.append(checkpoint(self.ff,value,use_reentrant=True))
            else:
                outputs.append(self.ff(value))
        ff_output=torch.cat(outputs,dim=1)
    else:
        ff_output=self.ff(norm_hidden_states)
    return residual+gather(gate_mlp)*ff_output


def install_stable_modulation(transformer):
    blocks=list(transformer.transformer_blocks)
    for block in blocks:
        if not hasattr(block,'_original_modulation_forward'):
            block._original_modulation_forward=block.forward
            block.forward=types.MethodType(stable_block_forward,block)
        block._segmented_modulation_enabled=True
        block._modulation_segments=None
    return blocks


def make_training_model(transformer,stable=False):
    from h3_fixed_chunk_contract import FixedTrainingModel
    if not stable:
        return FixedTrainingModel(transformer)
    class StableModel(FixedTrainingModel):
        def forward(self,batch):
            inputs=batch['inputs']
            indices=inputs['timestep_indices']*3+inputs['token_tags']
            segments=segments_for(indices)
            for block in self._stable_blocks:
                block._modulation_segments=segments
            return super().forward(batch)
    blocks=install_stable_modulation(transformer)
    model=StableModel(transformer)
    # Plain references, not another registered ModuleList/state_dict prefix.
    model._stable_blocks=blocks
    return model
