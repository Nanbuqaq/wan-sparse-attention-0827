"""Pack selected archive runs directly into contiguous head-major CPU storage."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping

import torch

from .staging import PinnedStagingPool, StagingLease
from .transfer_plan import TransferPlan


@dataclass
class PackedArchiveRuns:
    key: torch.Tensor  # [B,H,P,D], truly contiguous within a head
    value: torch.Tensor
    lease: StagingLease | None
    cpu_prepare_s: float
    cpu_allocate_pin_s: float
    cpu_pack_s: float
    source_slice_count: int


def pack_archive_runs(frames: Mapping, plan: TransferPlan, *, pin_memory: bool,
                      pool: PinnedStagingPool | None = None,
                      fused: bool = False) -> PackedArchiveRuns:
    """Split runs at frame allocations and copy just selected physical tokens.

    Source [B,T,H,D] slices may be strided. CPU copy packs them into contiguous
    [B,H,P,D]; only the latter is advertised as a contiguous H2D payload.
    """
    started = time.perf_counter()
    if bool(plan.resident_logical_mask.any()):
        raise ValueError('partial residency requires cache composition')
    selected_frames = [frames[frame] for frame in plan.candidate_frame_ids]
    first = selected_frames[0]
    batch, heads, width = plan.physical_source_offsets.shape
    dim, dtype = first.key.shape[-1], first.key.dtype
    shape = (batch, heads, width, dim)
    if any(item.key.shape != (batch, plan.frame_tokens, heads, dim)
           or item.value.shape != item.key.shape or item.key.dtype != dtype
           or item.value.dtype != dtype or item.key.device.type != 'cpu'
           or item.value.device.type != 'cpu' for item in selected_frames):
        raise ValueError('archive frames must match transfer geometry and CPU dtype')
    prepare_s = time.perf_counter() - started
    allocate_start = time.perf_counter()
    lease = pool.acquire(shape, dtype, fused=fused) if pool else None
    if lease:
        key, value = lease.key, lease.value
    else:
        key = torch.empty(shape, dtype=dtype, pin_memory=pin_memory)
        value = torch.empty_like(key, pin_memory=pin_memory)
    allocate_s = time.perf_counter() - allocate_start
    pack_start = time.perf_counter()
    try:
        key.zero_()
        value.zero_()
        source_slices = 0
        for run in plan.source_runs:
            remaining = run.token_count
            source, destination = run.source_offset, run.destination_offset
            while remaining:
                frame_index, token = divmod(source, plan.frame_tokens)
                count = min(remaining, plan.frame_tokens - token)
                frame = selected_frames[frame_index]
                key[run.batch_index, run.head_index, destination:destination + count].copy_(
                    frame.key[run.batch_index, token:token + count, run.head_index])
                value[run.batch_index, run.head_index, destination:destination + count].copy_(
                    frame.value[run.batch_index, token:token + count, run.head_index])
                remaining -= count
                source += count
                destination += count
                source_slices += 1
    except BaseException:
        if lease:
            pool.release(lease)
        raise
    return PackedArchiveRuns(key, value, lease, prepare_s, allocate_s,
                             time.perf_counter() - pack_start, source_slices)
