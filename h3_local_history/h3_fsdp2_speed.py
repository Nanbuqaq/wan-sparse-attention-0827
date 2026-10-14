"""Switchable activation policies for paired H3 training-speed experiments."""
from contextlib import nullcontext

import torch
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    OffloadWrapper, checkpoint_wrapper, CheckpointImpl,
)
from train_h3_v2_fsdp2 import TrainingModel


class SpeedTrainingModel(TrainingModel):
    prune_empty_history = False

    def forward(self, batch):
        if not self.prune_empty_history:
            return super().forward(batch)
        if batch['history_rows'].shape[1] != 0:
            raise ValueError('empty-history pruning requires all ranks to have empty history')
        from h3_spatial_history_flow import velocity_loss
        inputs = dict(batch['inputs'])
        inputs['state_hidden_states'] = batch['history_rows'].new_empty(
            (batch['history_rows'].shape[0], 0, self.projection.out_features), dtype=torch.bfloat16)
        video, audio = self.transformer(**inputs)
        return velocity_loss(video, audio, batch)


class SwitchableOffloadWrapper(OffloadWrapper):
    enabled = True

    def forward(self, *args, **kwargs):
        context = torch.autograd.graph.save_on_cpu(pin_memory=True) if self.enabled else nullcontext()
        with context:
            return self._checkpoint_wrapped_module(*args, **kwargs)


def shard_speed_model(model, mesh, reshard_after_forward=True):
    policy = MixedPrecisionPolicy(cast_forward_inputs=False)
    model.transformer.gradient_checkpointing = False
    masters, wrappers, blocks = set(), [], list(model.transformer.transformer_blocks)
    for i, block in enumerate(blocks):
        ignored = {p for p in block.parameters() if p.dtype != torch.bfloat16}
        masters.update(ignored)
        wrapped = SwitchableOffloadWrapper(checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT))
        model.transformer.transformer_blocks[i] = wrapped
        fully_shard(wrapped, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward, ignored_params=ignored)
        wrappers.append(wrapped)
    for name in ('proj_in', 'audio_proj_in', 'time_embedder', 'proj_out', 'audio_proj_out'):
        fully_shard(getattr(model.transformer, name), mesh=mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward)
    fully_shard(model.projection, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard_after_forward, ignored_params=masters)
    return model, blocks, wrappers


VARIANTS = {
    'baseline': dict(query_tile=8192, ff_tile=8192, inner_checkpoint=True, offload_every=1),
    'no_inner_16k': dict(query_tile=16384, ff_tile=16384, inner_checkpoint=False, offload_every=1),
    'no_inner_32k': dict(query_tile=32768, ff_tile=32768, inner_checkpoint=False, offload_every=1),
    'half_offload_32k': dict(query_tile=32768, ff_tile=32768, inner_checkpoint=False, offload_every=2),
    'empty_pruned': dict(query_tile=8192, ff_tile=8192, inner_checkpoint=True, offload_every=1, prune_empty_history=True),
    'empty_pruned_32k': dict(query_tile=32768, ff_tile=32768, inner_checkpoint=False, offload_every=1, prune_empty_history=True),
    'no_inner_8k': dict(query_tile=8192, ff_tile=8192, inner_checkpoint=False, offload_every=1),
    'ff_only_8k': dict(query_tile=8192, ff_tile=8192, inner_checkpoint=True, ff_checkpoint=False, offload_every=1),
    'half_offload_8k': dict(query_tile=8192, ff_tile=8192, inner_checkpoint=True, offload_every=2),
}


def apply_variant(name, blocks, wrappers, processors):
    cfg = VARIANTS[name]
    for i, (block, wrapper, processor) in enumerate(zip(blocks, wrappers, processors)):
        block.ff_chunk_size = cfg['ff_tile']
        block.ff_chunk_checkpoint = cfg.get('ff_checkpoint',cfg['inner_checkpoint'])
        processor.query_chunk_size = cfg['query_tile']
        processor.query_checkpoint = cfg['inner_checkpoint']
        wrapper.enabled = i % cfg['offload_every'] == 0
    return cfg
