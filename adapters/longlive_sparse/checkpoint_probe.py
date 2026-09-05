"""In-memory, development-only causal forks of the pinned avg-pool RAG loop.

The no-intervention suffix MUST match uninterrupted generation bitwise before
any fork is interpreted. This is not a production resume/checkpoint format.
CPU archive tensors are shared read-only; GPU mutable caches are snapshotted.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import random

import torch

from .stats import SparseRunStats


@dataclass
class FrozenTensor:
    value: torch.Tensor
    device: str


def freeze_tree(value):
    if isinstance(value, torch.Tensor):
        return FrozenTensor(value.detach().cpu().clone(), str(value.device))
    if isinstance(value, dict):
        return {k: freeze_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [freeze_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(freeze_tree(v) for v in value)
    return copy.deepcopy(value)


def thaw_tree(value):
    if isinstance(value, FrozenTensor):
        return value.value.to(value.device).clone()
    if isinstance(value, dict):
        return {k: thaw_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [thaw_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(thaw_tree(v) for v in value)
    return copy.deepcopy(value)


def capture_checkpoint(pipeline, keywords):
    if pipeline.compression_method != 'avg_pool':
        raise ValueError('only audited avg_pool retrieval is supported')
    archive = pipeline.sparse_history_archive
    # cpu_*_frames are immutable after commit; copying every raw CPU KV is
    # unnecessary. Keep list containers independent for later append operations.
    caches = []
    for cache in pipeline.kv_cache1:
        caches.append({key: list(value) if key in ('cpu_k_frames', 'cpu_v_frames')
                       else freeze_tree(value) for key, value in cache.items()})
    return {'current_start': int(keywords['current_start']),
            'conditional_dict': freeze_tree(keywords['conditional_dict']),
            'first_memory_indices': freeze_tree(keywords['memory_indices']),
            'kv_cache': caches, 'crossattn_cache': freeze_tree(pipeline.crossattn_cache),
            'descriptors': freeze_tree(pipeline.latent_descriptors),
            'archive_layers': {layer: dict(frames) for layer, frames in archive._layers.items()},
            'archive_versions': (archive._storage_version, dict(archive._layer_storage_versions),
                                 dict(archive._frame_storage_versions)),
            'rng_cpu': torch.get_rng_state().clone(),
            'rng_cuda': torch.cuda.get_rng_state().clone()}


def restore_checkpoint(pipeline, snapshot):
    archive = pipeline.sparse_history_archive
    archive.clear_frames()
    archive._layers = {layer: dict(frames) for layer, frames in snapshot['archive_layers'].items()}
    version, layers, frames = snapshot['archive_versions']
    archive._storage_version = version
    archive._layer_storage_versions = dict(layers)
    archive._frame_storage_versions = dict(frames)
    archive.stats = SparseRunStats(method=pipeline.sparse_history_config.method)
    pipeline.kv_cache1 = [{key: list(value) if key in ('cpu_k_frames', 'cpu_v_frames')
                          else thaw_tree(value) for key, value in cache.items()}
                         for cache in snapshot['kv_cache']]
    pipeline.crossattn_cache = thaw_tree(snapshot['crossattn_cache'])
    pipeline.latent_descriptors = thaw_tree(snapshot['descriptors'])
    for module in pipeline.sparse_history_modules:
        module.clear_selection_cache()
        module.clear_capture_state()
    if pipeline.history_union_cache is not None:
        pipeline.history_union_cache.reset()
    torch.set_rng_state(snapshot['rng_cpu'])
    torch.cuda.set_rng_state(snapshot['rng_cuda'])


def choose_pulse_indices(reference, *, eligible_count: int, policy: str, seed: int):
    if reference is None or reference.shape[0] != 1:
        raise ValueError('pulse requires nonempty batch-one historical retrieval')
    count = reference.shape[1]
    if not 0 < count <= eligible_count:
        raise ValueError('invalid eligible/count geometry')
    if policy == 'reference':
        return reference.clone()
    if policy == 'oldest':
        ids = list(range(count))
    elif policy == 'newest':
        ids = list(range(eligible_count - count, eligible_count))
    elif policy == 'random':
        ids = random.Random(seed).sample(range(eligible_count), count)
    else:
        raise ValueError(f'unknown pulse policy {policy}')
    return torch.tensor([ids], device=reference.device, dtype=reference.dtype)


def retrieve_frames(pipeline, *, device):
    model = pipeline.args.model_kwargs
    sink = int(getattr(model, 'sink_size', 0))
    excluded = int(getattr(model, 'recent_exclude', 0))
    size = int(getattr(model, 'memory_size', 0))
    eligible = max(0, len(pipeline.kv_cache1[0].get('cpu_k_frames', [])) - excluded)
    if size <= 0 or not eligible or not pipeline.latent_descriptors:
        return None
    keys = torch.stack([pipeline.latent_descriptors[sink + i] for i in range(eligible)], dim=1)
    query = pipeline.latent_descriptors[-1].unsqueeze(1)
    query = query / (query.norm(dim=-1, keepdim=True) + 1e-8)
    keys = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
    sims = torch.bmm(keys, query.transpose(1, 2)).squeeze(-1)
    return torch.topk(sims, k=min(size, eligible), dim=-1).indices.to(device)


@torch.inference_mode()
def replay_suffix(pipeline, snapshot, noise, *, chunks: int, policy: str, pulse_seed: int):
    restore_checkpoint(pipeline, snapshot)
    frame_tokens = pipeline.frame_seq_length
    start_frame = snapshot['current_start'] // frame_tokens
    block = pipeline.num_frame_per_block
    if start_frame + chunks * block > noise.shape[1]:
        raise ValueError('suffix extends beyond saved noise')
    conditional = thaw_tree(snapshot['conditional_dict'])
    outputs, retrievals = [], []
    steps = pipeline.denoising_step_list
    for offset in range(chunks):
        current_frame = start_frame + offset * block
        current = noise[:, current_frame:current_frame + block]
        memory = (thaw_tree(snapshot['first_memory_indices']) if offset == 0
                  else retrieve_frames(pipeline, device=noise.device))
        if offset == 0:
            excluded = int(getattr(pipeline.args.model_kwargs, 'recent_exclude', 0))
            eligible = max(0, len(pipeline.kv_cache1[0]['cpu_k_frames']) - excluded)
            memory = choose_pulse_indices(memory, eligible_count=eligible, policy=policy, seed=pulse_seed)
        retrievals.append({'frame': current_frame, 'pool_indices': memory.cpu().tolist() if memory is not None else None,
                           'intervention_active': offset == 0 and policy != 'reference'})
        for index, step in enumerate(steps):
            timestep = torch.ones([noise.shape[0], block], device=noise.device, dtype=torch.int64) * step
            _, denoised = pipeline.generator(noisy_image_or_video=current, conditional_dict=conditional,
                timestep=timestep, kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
                current_start=current_frame * frame_tokens, memory_indices=memory)
            if index < len(steps) - 1:
                current = pipeline.scheduler.add_noise(denoised.flatten(0, 1),
                    torch.randn_like(denoised.flatten(0, 1)),
                    steps[index + 1] * torch.ones([noise.shape[0] * block], device=noise.device, dtype=torch.long)
                ).unflatten(0, denoised.shape[:2])
        outputs.append(denoised.detach().cpu())
        for frame in range(block):
            pipeline.latent_descriptors.append(denoised[:, frame].mean(dim=(-2, -1)).detach())
        pipeline.generator(noisy_image_or_video=denoised, conditional_dict=conditional,
            timestep=torch.ones_like(timestep) * getattr(pipeline.args, 'context_noise', 0.),
            kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
            current_start=current_frame * frame_tokens, memory_indices=memory)
    return torch.cat(outputs, dim=1), retrievals
