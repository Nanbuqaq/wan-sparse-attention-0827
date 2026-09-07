"""Finite-buffer CPU producer -> H2D -> GPU-consumer pipeline.

Frame tensors remain in their original CPU [T,H,D] storage. No complete
candidate concatenation or full-history GPU shadow is constructed. Callers
provide an already-known physical page order and a GPU consumer; this module
does not predict routes or change their logical selection.
"""
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
import threading
import time

import torch


@dataclass(frozen=True)
class FramePage:
    frame: int
    start: int
    count: int


def frame_pages(lengths, page_tokens):
    if page_tokens < 1 or not lengths or any(n < 1 for n in lengths):
        raise ValueError('positive frame lengths and page size required')
    return tuple(FramePage(f, start, min(page_tokens, n-start))
                 for f, n in enumerate(lengths) for start in range(0, n, page_tokens))


@dataclass
class _Job:
    order: tuple[int, ...]
    origin: object
    free: Queue
    ready: Queue
    stop: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    errors: list = field(default_factory=list)
    metrics: dict = field(default_factory=lambda: dict(CPU_pack_s=0., CPU_host_slot_wait_s=0.,
        CPU_device_slot_queue_wait_s=0., H2D_payload_bytes=0, H2D_padding_bytes=0, H2D_copy_calls=0))


def _get(queue, job):
    while not job.stop.is_set():
        try:
            return queue.get(timeout=.1)
        except Empty:
            if job.errors:
                raise RuntimeError('page producer failed') from job.errors[0]
            if job.finished.is_set():
                raise RuntimeError('page producer ended before requested page')
    raise RuntimeError('page pipeline cancelled')


def _put(queue, item, job):
    while not job.stop.is_set():
        try:
            queue.put(item, timeout=.1)
            return
        except Full:
            pass
    raise RuntimeError('page pipeline cancelled')


class BoundedPagePipeline:
    def __init__(self, frames, *, page_tokens, capacity, device, host_slots=2):
        started = time.perf_counter()
        if capacity < 1 or host_slots < 1 or not frames:
            raise ValueError('positive capacity and host slots required')
        self.frames = tuple(frames)
        key = frames[0][0]
        if key.ndim != 3 or key.device.type != 'cpu':
            raise ValueError('archive frames must be CPU [T,H,D] tensors')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('this pipeline requires real CUDA')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.dtype, self.heads, self.dim = key.dtype, key.shape[1], key.shape[2]
        for k, v in frames:
            if (k.shape != v.shape or k.ndim != 3 or k.shape[1:] != key.shape[1:]
                    or k.dtype != self.dtype or v.dtype != self.dtype
                    or k.device.type != 'cpu' or v.device.type != 'cpu'):
                raise ValueError('archive shape/dtype/device mismatch')
        self.pages = frame_pages([k.shape[0] for k, _ in frames], page_tokens)
        self.page_tokens, self.capacity = page_tokens, capacity
        shape = (2, page_tokens, self.heads, self.dim)
        self.host = [torch.empty(shape, dtype=self.dtype, pin_memory=True) for _ in range(host_slots)]
        self.host_done = [None] * host_slots
        self.gpu = torch.empty((capacity, *shape), dtype=self.dtype, device=self.device)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.commands = Queue(maxsize=1)
        self.active = threading.Lock()
        self.closed = False
        self.thread = threading.Thread(target=self._worker, name='longlive-page-producer', daemon=True)
        self.thread.start()
        self.startup_wall_s = time.perf_counter() - started

    def _stage(self, job, ordinal, page_id, stream):
        begin = time.perf_counter()
        slot, consumed = _get(job.free, job)
        job.metrics['CPU_device_slot_queue_wait_s'] += time.perf_counter()-begin
        host_slot = ordinal % len(self.host)
        if self.host_done[host_slot] is not None:
            begin = time.perf_counter()
            self.host_done[host_slot].synchronize()
            job.metrics['CPU_host_slot_wait_s'] += time.perf_counter()-begin
        page = self.pages[page_id]
        key, value = self.frames[page.frame]
        begin = time.perf_counter()
        staging = self.host[host_slot]
        if page.count < self.page_tokens:
            staging[:, page.count:].zero_()
        staging[0, :page.count].copy_(key[page.start:page.start+page.count])
        staging[1, :page.count].copy_(value[page.start:page.start+page.count])
        job.metrics['CPU_pack_s'] += time.perf_counter()-begin
        with torch.cuda.stream(stream):
            if consumed is not None:
                # Event has already been recorded by the consumer before the
                # slot is put back in the free queue; never wait on unrecorded events.
                stream.wait_event(consumed)
            self.gpu[slot].copy_(staging, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record()
        self.host_done[host_slot] = ready
        token_bytes = 2*self.heads*self.dim*key.element_size()
        job.metrics['H2D_payload_bytes'] += page.count*token_bytes
        job.metrics['H2D_padding_bytes'] += (self.page_tokens-page.count)*token_bytes
        job.metrics['H2D_copy_calls'] += 1
        return ordinal, page_id, slot, page.count, ready

    def _worker(self):
        with torch.cuda.device(self.device), torch.inference_mode():
            while True:
                job = self.commands.get()
                if job is None:
                    return
                try:
                    self.copy_stream.wait_event(job.origin)
                    for ordinal, page in enumerate(job.order):
                        item = self._stage(job, ordinal, page, self.copy_stream)
                        _put(job.ready, item, job)
                except BaseException as error:
                    job.errors.append(error)
                finally:
                    job.finished.set()

    def run(self, order, consumer, *, mode='producer', initialize=None, finalize=None):
        if self.closed or not self.active.acquire(blocking=False):
            raise RuntimeError('pipeline closed or concurrent invocation')
        job = None
        try:
            order = tuple(int(i) for i in order)
            if not order or min(order) < 0 or max(order) >= len(self.pages):
                raise ValueError('invalid page order')
            if mode not in ('serial', 'same_thread_async', 'producer'):
                raise ValueError('unknown page scheduling mode')
            origin = torch.cuda.Event()
            origin.record(torch.cuda.current_stream(self.device))
            job = _Job(order, origin, Queue(maxsize=self.capacity), Queue(maxsize=self.capacity))
            for slot in range(self.capacity):
                job.free.put((slot, None))
            self.compute_stream.wait_event(origin)
            self.copy_stream.wait_event(origin)
            started = time.perf_counter()
            if mode == 'producer':
                self.commands.put(job)
            with torch.cuda.stream(self.compute_stream):
                if initialize is not None:
                    initialize()
            consumer_wait = 0.
            for ordinal, page_id in enumerate(order):
                if mode == 'producer':
                    begin = time.perf_counter()
                    item = _get(job.ready, job)
                    consumer_wait += time.perf_counter()-begin
                else:
                    stream = self.compute_stream if mode == 'serial' else self.copy_stream
                    item = self._stage(job, ordinal, page_id, stream)
                actual_ordinal, actual_page, slot, count, ready = item
                if (actual_ordinal, actual_page) != (ordinal, page_id):
                    raise RuntimeError('producer changed page order')
                with torch.cuda.stream(self.compute_stream):
                    self.compute_stream.wait_event(ready)
                    consumer(page_id, self.gpu[slot, 0, :count], self.gpu[slot, 1, :count])
                    consumed = torch.cuda.Event()
                    consumed.record()
                job.free.put((slot, consumed))
            with torch.cuda.stream(self.compute_stream):
                output = finalize() if finalize is not None else None
                complete = torch.cuda.Event()
                complete.record()
            complete.synchronize()  # Full replay timing and storage lifetime fence.
            if mode == 'producer':
                if not job.finished.wait(timeout=30):
                    raise RuntimeError('producer did not finish after all pages were consumed')
                if job.errors:
                    raise RuntimeError('page producer failed') from job.errors[0]
            return output, dict(job.metrics, full_wall_s=time.perf_counter()-started,
                CPU_consumer_queue_wait_s=consumer_wait, mode=mode,
                GPU_KV_capacity_bytes=self.gpu.numel()*self.gpu.element_size(),
                host_pinned_bytes=sum(x.numel()*x.element_size() for x in self.host),
                producer_startup_wall_s=self.startup_wall_s,
                actual_overlap_requires_Nsight=True, upstream_candidate_concatenation=False)
        finally:
            if job is not None and mode == 'producer' and not job.finished.is_set():
                job.stop.set()
                job.finished.wait(timeout=30)
            self.active.release()

    def close(self):
        if self.closed:
            return
        if not self.active.acquire(blocking=False):
            raise RuntimeError('cannot close an active pipeline')
        try:
            self.compute_stream.synchronize()
            self.copy_stream.synchronize()
            self.commands.put(None)
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise RuntimeError('producer did not close')
            self.closed = True
        finally:
            self.active.release()
