"""Offline-only Tether mask addressing and compact region/age bias.

The source-compatible variant reproduces released mask-addressing semantics,
not the complete original pipeline. The aligned variant is a mechanism teacher.
Neither class/function is an online selector or a physical KV pruning method.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import torch

from .attention_bias import AttentionBiasPlan
from .tethermem import solve_context_weight


@lru_cache(maxsize=4)
def load_oracle_masks(path, mask_sha, reference_video_sha):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != mask_sha:
        raise ValueError('oracle mask SHA differs from frozen method identity')
    producer = json.loads(path.with_name('result.json').read_text())
    if producer['status'] != 'pass' or producer['video_sha256'] != reference_video_sha:
        raise ValueError('oracle producer or Dense reference mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('causal_online') is not False or payload.get('mask_scope') != 'offline_dense_reference_teacher':
        raise ValueError('explicit offline Dense/SAM2 masks required')
    return payload, producer


def build_oracle_bias(route, masks, *, timeline, current_latent, sink_frames, cpu_pool_frames,
                       target_average=.25, age_decay_floor=.05):
    if timeline not in ('source_compatible_addressing', 'aligned_latent_anchors'):
        raise ValueError('unknown oracle temporal addressing')
    if route.union_frame_ids.shape[0] != 1 or route.query_tokens % 1560 or route.history_pair_density != 1.:
        raise ValueError('oracle requires batch-one complete retrieved history and full query frames')
    qframes = route.query_tokens//1560
    pixel = masks['pixel_patch_masks'] > .5
    latent = masks['latent_anchor_masks'] > .5
    if current_latent < 0 or current_latent+qframes > len(latent):
        raise ValueError('reference video does not cover oracle query')
    if timeline == 'source_compatible_addressing':
        qmask = pixel[min(current_latent, len(pixel)-1)].flatten().repeat(qframes)
        # Public source uses CPU pool indices without global sink/time mapping.
        frame_lookup = route.union_frame_ids-sink_frames
        history_masks = pixel
    else:
        qmask = latent[current_latent:current_latent+qframes].flatten()
        frame_lookup = route.union_frame_ids
        history_masks = latent
    if bool((route.union_frame_ids < sink_frames).any()) or bool((route.union_token_ids < 0).any()):
        raise ValueError('full oracle history may not contain padding or sink coordinates')
    frame_lookup = frame_lookup.clamp(0, len(history_masks)-1)
    keymask = history_masks.flatten(1)[frame_lookup, route.union_token_ids]
    pool_indices = route.union_frame_ids-sink_frames
    age = (pool_indices.float()/max(min(cpu_pool_frames, 120), 1)).clamp_min(age_decay_floor)
    ratio = float(qmask.float().mean())
    weight = solve_context_weight(ratio, 1., target_average)
    return AttentionBiasPlan(('identity', 'scene'),
        torch.stack((qmask.float(), (~qmask).float()), -1).unsqueeze(0),
        torch.stack((keymask.float(), (~keymask).float()), -1), age,
        mode='tethermem_oracle_mask_teacher', metadata={'context_weight': weight,
            'mask_timeline': timeline, 'query_subject_fraction': ratio,
            'target_average': target_average, 'achieved_average': ratio+(1-ratio)*weight,
            'target_unattainable_by_context_only': ratio > target_average,
            'online_method': False, 'physical_history_density': 1.,
            'source_compatible_addressing_is_not_full_pipeline_bitwise_reproduction': True})
