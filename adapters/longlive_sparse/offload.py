"""Bounded pinned D2H tickets whose committed archive storage is pageable.

The runtime initially completes each ticket synchronously. The split interface
also permits explicit overlap replay, but does not itself prove overlap. Source
GPU tensors must remain immutable until completion (or request a snapshot).
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import torch
from .staging import PinnedStagingPool, StagingLease
from .profiling import profiled


@dataclass
class OffloadTicket:
    owner: object
    lease: StagingLease
    key_source: torch.Tensor
    value_source: torch.Tensor
    archive_key: torch.Tensor
    archive_value: torch.Tensor
    start_event: object
    ready_event: object
    started: float
    allocate_s: float
    snapshot_source: bool
    completed: bool = False


class ArchiveOffloadStager:
    def __init__(self, pool: PinnedStagingPool):
        self.pool=pool
        self.streams={}
        self.completed=0
        self.payload_bytes=0
        self.cpu_commit_s=0.

    @profiled('archive/launch_bounded_d2h')
    def launch(self, key, value, *, snapshot_source=False):
        started=time.perf_counter()
        if key.shape!=value.shape or key.dtype!=value.dtype or key.device!=value.device:
            raise ValueError('offload K/V geometry, dtype and device must match')
        if key.is_cuda and not self.pool.pin_memory:
            raise ValueError('GPU D2H staging requires a pinned pool')
        lease=self.pool.acquire(tuple(key.shape),key.dtype,fused=False)
        stream=None
        try:
            archive_key=torch.empty(key.shape,dtype=key.dtype,device='cpu',pin_memory=False)
            archive_value=torch.empty_like(archive_key,pin_memory=False)
            allocate_s=time.perf_counter()-started
            if snapshot_source:
                key,value=key.clone(),value.clone()
            begin=ready=None
            if key.is_cuda:
                stream=self.streams.setdefault(str(key.device),None)
                if stream is None:
                    stream=torch.cuda.Stream(device=key.device)
                    self.streams[str(key.device)]=stream
                stream.wait_stream(torch.cuda.current_stream(key.device))
                begin=torch.cuda.Event(enable_timing=True)
                ready=torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    begin.record()
                    lease.key.copy_(key,non_blocking=True)
                    lease.value.copy_(value,non_blocking=True)
                    ready.record()
                key.record_stream(stream)
                value.record_stream(stream)
            else:
                lease.key.copy_(key)
                lease.value.copy_(value)
            return OffloadTicket(self,lease,key,value,archive_key,archive_value,begin,ready,
                                  started,allocate_s,snapshot_source)
        except BaseException:
            # Never recycle a CPU destination while a submitted DMA may still
            # be writing it, even on a partially failed launch.
            try:
                if stream is not None:
                    stream.synchronize()
            finally:
                self.pool.release(lease)
            raise

    @profiled('archive/commit_pageable_storage')
    def complete(self, ticket):
        if ticket.owner is not self or ticket.completed:
            raise ValueError('offload ticket is foreign or already completed')
        wait_start=time.perf_counter()
        if ticket.ready_event is not None:
            ticket.ready_event.synchronize()
        waited=time.perf_counter()-wait_start
        commit_start=time.perf_counter()
        try:
            ticket.archive_key.copy_(ticket.lease.key)
            ticket.archive_value.copy_(ticket.lease.value)
        finally:
            self.pool.release(ticket.lease)
            ticket.completed=True
        commit_s=time.perf_counter()-commit_start
        payload=(ticket.archive_key.numel()+ticket.archive_value.numel())*ticket.archive_key.element_size()
        self.completed+=1
        self.payload_bytes+=payload
        self.cpu_commit_s+=commit_s
        metrics={'payload_bytes':payload,'torch_d2h_copy_calls':2,'pool_reused':ticket.lease.reused,
            'allocation_s':ticket.allocate_s,'exposed_ready_wait_s':waited,'cpu_commit_s':commit_s,
            'launch_to_commit_s':time.perf_counter()-ticket.started,
            'd2h_device_s':ticket.start_event.elapsed_time(ticket.ready_event)/1000 if ticket.ready_event is not None else None,
            'archive_is_pinned':ticket.archive_key.is_pinned() or ticket.archive_value.is_pinned(),
            'gpu_snapshot_requested':ticket.snapshot_source,'runtime_overlap_claim':False}
        return ticket.archive_key,ticket.archive_value,metrics

    def as_dict(self):
        return {'completed_tickets':self.completed,'payload_bytes':self.payload_bytes,
                'cpu_commit_s':self.cpu_commit_s,'storage':'pageable','runtime_overlap_claim':False}
