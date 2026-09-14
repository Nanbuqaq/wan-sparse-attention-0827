"""Bounded asynchronous snapshots with per-layer cache-mutation fences.

Two pinned layer slots feed compact pageable CPU archives. Native KV is
borrowed, never cloned wholesale, and cannot be overwritten until its D2H
copy completes. This changes scheduling only, not source data or selection.
"""
import queue
import threading
import time
import traceback

import torch

from .immutable_source_reader import SideArchive, ImmutableSourceReader
from .layer_stream_source_reader import LayerStreamSourceReader
from .native_scene_admission import SceneDescriptor
from .history_cache import tensor_sha256


class SnapshotJob:
    def __init__(self,source,destination,record,source_ready,guard_needed):
        self.source=source;self.destination=destination;self.record=record
        self.source_ready=source_ready;self.guard_needed=guard_needed
        self.events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in source]
        self.queued=[threading.Event() for _ in source]
        self.guarded=[False]*len(source);self.done=threading.Event();self.error=None

    def check(self):
        if self.error is not None:raise RuntimeError('archive worker failed: '+self.error)

    def wait(self):
        if not self.done.wait(120):raise RuntimeError('archive producer did not finish within120s')
        self.check()


class StagedSceneArchive(SideArchive):
    def __init__(self,pipe,*,serial=False,**kwargs):
        super().__init__(pipe,**kwargs)
        self.serial=serial
        self.active_job=None;self.worker=None;self.tasks=queue.Queue(maxsize=1)
        self.buffers=[];self.copy_stream=None;self.reports=[];self.fences=[];self.closed=False
        self.stats=dict(pinned_slots=2,pinned_limit_bytes=256*1024**2,pinned_peak_bytes=0,
            staging_initialization_host_s=0.,CPU_archive_allocation_host_s=0.,
            previous_job_wait_host_s=0.,consumer_queue_wait_host_s=0.,
            demand_ready_wait_host_s=0.,close_wait_host_s=0.,protected_global_fence_bypasses=0,
            borrowed_GPU_views_extra_owned_storage_bytes=0)

    def _initialize(self,example):
        if self.worker is not None:return
        began=time.perf_counter();shape=(2,*example.shape)
        needed=2*2*example.numel()*example.element_size()
        if needed>self.stats['pinned_limit_bytes']:raise RuntimeError('two pinned layer slots exceed256MiB')
        self.buffers=[torch.empty(shape,dtype=example.dtype,pin_memory=True) for _ in range(2)]
        self.stats['pinned_peak_bytes']=sum(x.untyped_storage().nbytes() for x in self.buffers)
        self.copy_stream=torch.cuda.Stream(device=example.device)
        self.device=example.device
        self.worker=threading.Thread(target=self._worker,name='native-archive-copy-worker',daemon=True)
        self.worker.start();self.stats['staging_initialization_host_s']+=time.perf_counter()-began

    def _worker(self):
        with torch.inference_mode(),torch.cuda.device(self.device):
            while True:
                job=self.tasks.get()
                if job is None:return
                began=time.perf_counter();pending={};copy_rows=[]
                def drain(slot):
                    layer=pending.pop(slot);ready=job.events[layer][1]
                    started=time.perf_counter();ready.synchronize();readiness=time.perf_counter()-started
                    started=time.perf_counter()
                    for i,destination in enumerate(job.destination[layer]):destination.copy_(self.buffers[slot][i])
                    copy_rows.append(dict(layer=layer,CPU_copy_host_s=time.perf_counter()-started,
                        worker_DMA_readiness_wait_host_s=readiness,
                        D2H_stream_ms=job.events[layer][0].elapsed_time(ready)))
                try:
                    with torch.cuda.stream(self.copy_stream):
                        self.copy_stream.wait_event(job.source_ready)
                        for layer,pair in enumerate(job.source):
                            slot=layer%2
                            if slot in pending:drain(slot)
                            start,end=job.events[layer];start.record(self.copy_stream)
                            for i,source in enumerate(pair):self.buffers[slot][i].copy_(source,non_blocking=True)
                            end.record(self.copy_stream);pending[slot]=layer
                            # Waiting on an unrecorded CUDA event would be a no-op.
                            # Publish readiness only after its record is enqueued.
                            job.queued[layer].set()
                        for slot in list(pending):drain(slot)
                    job.record.update(ready=True,producer_wall_s=time.perf_counter()-began,
                        layers=sorted(copy_rows,key=lambda row:row['layer']))
                except BaseException:
                    job.error=traceback.format_exc();job.record.update(ready=False,error=job.error)
                    for flag in job.queued:flag.set()
                finally:
                    # Reports/jobs must not retain evicted raw CPU banks or the
                    # whole live GPU cache through view references.
                    job.source=[];job.destination=[];job.done.set()

    def before_layer_mutation(self,layer):
        job=self.active_job
        if job is None:return
        job.check()
        if job.done.is_set() or job.guarded[layer]:return
        if not job.guard_needed:
            # Under this qualified immutable-reader path, the initial global
            # slots are never rewritten after their completed first chunk.
            job.guarded[layer]=True;self.stats['protected_global_fence_bypasses']+=1;return
        began=time.perf_counter()
        if not job.queued[layer].wait(120):raise RuntimeError('archive layer copy was not enqueued within120s')
        self.stats['consumer_queue_wait_host_s']+=time.perf_counter()-began
        job.check()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        stream=torch.cuda.current_stream(self.device)
        start.record(stream);stream.wait_event(job.events[layer][1]);end.record(stream)
        self.fences.append((start,end));job.guarded[layer]=True
        if len(self.fences)>61440:raise RuntimeError('finite archive fence bound exceeded')

    def _archive_last_scene(self,current_frame):
        if self.closed:raise RuntimeError('archive is closed')
        last=self.last_commit
        if last is None or last['end']!=current_frame:raise RuntimeError('only a just-closed clean scene may be archived')
        began=time.perf_counter()
        if self.active_job is not None:
            started=time.perf_counter();self.active_job.wait()
            self.stats['previous_job_wait_host_s']+=time.perf_counter()-started
        caches=self.pipe.kv_cache_pos;n=8*self.pipe.frame_seq_length;source=[];starts=[]
        for cache in caches:
            end=int(cache['local_end_index'])
            if int(cache['global_end_index'])!=current_frame*self.pipe.frame_seq_length or end<n:
                raise RuntimeError('source bounds differ from committed cache')
            starts.append(end-n);source.append((cache['k'][:,end-n:end],cache['v'][:,end-n:end]))
        if len(set(starts))!=1:raise RuntimeError('snapshot physical source ranges differ across layers')
        self._initialize(source[0][0])
        required=sum(t.numel()*t.element_size() for pair in source for t in pair)
        owned=required+last['prototype'].numel()*last['prototype'].element_size()
        if owned>self.budget:raise RuntimeError('one archive exceeds its CPU budget')
        while sum(b['owned_bytes'] for b in self.banks)+owned>self.budget:
            self.banks.pop(0);self.ledger['evicted_archives']+=1
        started=time.perf_counter()
        destination=[tuple(torch.empty(tuple(t.shape),dtype=t.dtype,device='cpu') for t in pair) for pair in source]
        self.stats['CPU_archive_allocation_host_s']+=time.perf_counter()-started
        self.version+=1;descriptor=SceneDescriptor(self.version,current_frame,last['phase'],last['prototype'])
        record=dict(archive_version=self.version,source_frames=list(range(current_frame-8,current_frame)),
            source_end=current_frame,source_phase=last['phase'],KV_bytes=required,
            condition_prototype_bytes=owned-required,unique_owned_raw_storage_bytes=required,
            logical_tensor_bytes=required,storage_count=2*len(source),ready=False)
        ready=torch.cuda.Event();ready.record(torch.cuda.current_stream(self.device))
        global_tokens=self.pipe.sink_size*self.pipe.frame_seq_length
        guard_needed=not (starts[0]>=0 and starts[0]+n<=global_tokens)
        job=SnapshotJob(source,destination,record,ready,guard_needed)
        self.banks.append(dict(descriptor=descriptor,kv=destination,owned_bytes=owned,copy_job=job))
        self.archives.append(record);self.reports.append(record);self.active_job=job
        self.ledger['archive_D2H_KV_bytes']+=required
        self.ledger['CPU_archive_peak_tensor_bytes']=max(self.ledger['CPU_archive_peak_tensor_bytes'],sum(b['owned_bytes'] for b in self.banks))
        self.tasks.put_nowait(job)
        if self.serial:job.wait()
        self.ledger['archive_wall_s']+=time.perf_counter()-began

    def wait_for_bank(self,bank):
        began=time.perf_counter();bank['copy_job'].wait()
        self.stats['demand_ready_wait_host_s']+=time.perf_counter()-began

    def close(self):
        if self.closed:return
        began=time.perf_counter()
        try:
            if self.active_job is not None:self.active_job.wait()
        finally:
            if self.worker is not None:
                self.tasks.put(None);self.worker.join(timeout=10)
                if self.worker.is_alive():raise RuntimeError('archive worker did not stop')
            self.closed=True
            self.stats['close_wait_host_s']+=time.perf_counter()-began

    def audit(self):
        if self.active_job is not None:self.active_job.wait()
        result=super().audit()
        if self.fences:torch.cuda.synchronize(self.device)
        result['staged_archive']=dict(**self.stats,producer_records=self.reports,serial_producer_control=self.serial,
            GPU_fence_stream_ms=sum(a.elapsed_time(b) for a,b in self.fences),fence_count=len(self.fences),
            archive_wall_s_scope='submission plus producer completion' if self.serial else 'submission host span; producer work and fences separately recorded',
            timing_scope='CPU, GPU stream and producer spans overlap; do not add them',
            raw_values_and_selection_unchanged=True,no_full_GPU_snapshot_copy=True,
            CPU_raw_budget_excludes_the_separately_bounded_pinned_pool=True)
        return result


class ArchiveStagingMixin:
    def __init__(self,*args,archive_staging=False,archive_serial=False,archive_digest=False,archive_readiness='generation',**kwargs):
        super().__init__(*args,**kwargs)
        if (self.source_policy!='full_once' or not self.source_archive_enabled
                or self.snapshot_window!='latest8' or self.source_backend!='concat'):
            raise ValueError('first archive experiment fixes a real latest8 source reader')
        if archive_readiness not in ('device','generation'):raise ValueError('unknown source readiness scope')
        self.archive_readiness=archive_readiness
        self.archive_staging=archive_staging;self.archive_digest=archive_digest;self.archive_fence_handles=[]
        if archive_staging:self.side_archive=StagedSceneArchive(self.pipe,serial=archive_serial,archive_budget=8*1024**3)

    def source_readiness_sync(self,device=None):
        if device is None:device=self.pipe.kv_cache_pos[0]['k'].device
        if self.archive_readiness=='generation':torch.cuda.current_stream(device).synchronize()
        else:super().source_readiness_sync(device)

    def attach(self):
        super().attach()
        if self.archive_staging:
            for i,block in enumerate(self.layers):
                self.archive_fence_handles.append(block.self_attn.register_forward_pre_hook(
                    lambda m,a,kw,index=i:self.side_archive.before_layer_mutation(index),with_kwargs=True))

    def _load_side(self,bank,frame,text):
        if self.archive_staging:self.side_archive.wait_for_bank(bank)
        return super()._load_side(bank,frame,text)

    def detach(self):
        try:
            if self.archive_staging:self.side_archive.close()
        finally:
            for handle in self.archive_fence_handles:handle.remove()
            self.archive_fence_handles=[];super().detach()

    def audit(self):
        result=super().audit()
        result['archive_source_readiness_scope']=self.archive_readiness
        if self.archive_digest:
            began=time.perf_counter()
            result['archive_digest_gate']=dict(records=[dict(version=b['descriptor'].archive_version,
                source_end=b['descriptor'].source_end,
                layers=[dict(K=tensor_sha256(k),V=tensor_sha256(v)) for k,v in b['kv']]) for b in self.side_archive.banks],
                all_retained_raw_tensors_hashed=True,not_selector_input=True)
            result['archive_digest_gate']['CPU_digest_host_s']=time.perf_counter()-began
        return result


class StagedResidentReader(ArchiveStagingMixin,ImmutableSourceReader):pass
class StagedLayerStreamReader(ArchiveStagingMixin,LayerStreamSourceReader):pass
