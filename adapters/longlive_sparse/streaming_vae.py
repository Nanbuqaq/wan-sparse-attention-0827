"""Continuous-cache VAE stream with bounded pinned output slots.

Only completed latent chunks may be submitted, in order. GPU decode dispatch
stays on the submitting thread; a completion thread drains ready CPU pixels.
This avoids concurrently mutating the VAE's Python temporal cache. Streaming
decode/event ordering follows the public LongLive2 design; this implementation
adds an explicit bounded slot pool and separates ready pixels from encoding.
"""
import contextlib
import math
from queue import Empty, Queue
import threading
import time

import torch
from .profiling import profiled


class StreamingVAEDecoder:
    def __init__(self, vae, *, device, dtype=torch.bfloat16, max_chunk_latents=3,
                 slots=2, pinned_budget_bytes=128*1024**2, collect=True, on_chunk=None):
        if slots < 1 or max_chunk_latents < 1 or pinned_budget_bytes < 1:
            raise ValueError('positive slot, chunk and pinned budgets required')
        if not collect and on_chunk is None:
            raise ValueError('a streaming sink is required when outputs are not collected')
        self.vae, self.device, self.dtype = vae, torch.device(device), dtype
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.is_cuda = self.device.type == 'cuda'
        self.stream = torch.cuda.Stream(device=self.device) if self.is_cuda else None
        self.max_chunk_latents, self.slots, self.budget = max_chunk_latents, slots, pinned_budget_bytes
        self.collect, self.on_chunk = collect, on_chunk
        self.free, self.jobs = Queue(maxsize=slots), Queue()
        for slot in range(slots):
            self.free.put(slot)
        self.buffers = None
        self.outputs, self.records = [], []
        self.error = None
        self.closed = False
        self.latents_submitted = self.pixels_submitted = 0
        self.producer_backpressure_s = 0.
        self.started = time.perf_counter()
        self.vae.model.clear_cache()
        scale_scope = torch.cuda.stream(self.stream) if self.is_cuda else contextlib.nullcontext()
        with scale_scope:
            # Constants are queued on their consuming stream once, not copied
            # with blocking .to() calls on every completed latent chunk.
            self.scale = [self.vae.mean.to(device=self.device, dtype=self.dtype),
                          1./self.vae.std.to(device=self.device, dtype=self.dtype)]
        self.worker = threading.Thread(target=self._drain, name='longlive-vae-completion', daemon=True)
        self.worker.start()

    def _drain(self):
        device_scope = torch.cuda.device(self.device) if self.is_cuda else contextlib.nullcontext()
        with device_scope, torch.inference_mode():
            try:
                while True:
                    job = self.jobs.get()
                    if job is None:
                        return
                    slot, view, ready, begin, record, snapshot, decoded = job
                    if ready is not None:
                        ready.synchronize()
                        record['decode_stream_span_s_not_service'] = begin.elapsed_time(ready)/1000
                    record['pixels_ready_s'] = time.perf_counter()-self.started
                    # Do not retain pinned storage in the collected video.
                    pixels = torch.empty_like(view, device='cpu', pin_memory=False).copy_(view)
                    pixels = pixels.permute(0, 2, 1, 3, 4)
                    if not bool(torch.isfinite(pixels).all()):
                        raise RuntimeError('nonfinite streaming VAE output')
                    if self.collect:
                        self.outputs.append(pixels)
                    if self.on_chunk is not None:
                        self.on_chunk(pixels, dict(record))
                    record['sink_completed_s'] = time.perf_counter()-self.started
                    self.records.append(record)
                    # snapshot/decoded references fence CUDA allocator lifetime
                    # through D2H completion before the next slot reuse.
                    del snapshot, decoded, view, pixels, job
                    self.free.put(slot)
            except BaseException as error:
                self.error = error

    @torch.inference_mode()
    @profiled('vae/stream_submit')
    def submit(self, latent, *, start_latent):
        if self.closed or self.error is not None:
            raise RuntimeError('streaming VAE closed or failed') from self.error
        if latent.ndim != 5 or latent.shape[0] != 1 or not 0 < latent.shape[1] <= self.max_chunk_latents:
            raise ValueError('submit one batch of bounded completed [B,T,C,H,W] latents')
        if int(start_latent) != self.latents_submitted:
            raise ValueError('completed latent chunks must arrive in order without gaps')
        begin_wait = time.perf_counter()
        while True:
            if self.error is not None:
                raise RuntimeError('streaming VAE failed while waiting for a slot') from self.error
            try:
                slot = self.free.get(timeout=.1)
                break
            except Empty:
                pass
        self.producer_backpressure_s += time.perf_counter()-begin_wait
        try:
            snapshot = latent.detach().to(device=self.device, dtype=self.dtype).clone()
            source_ready = torch.cuda.Event() if self.is_cuda else None
            if source_ready is not None:
                source_ready.record(torch.cuda.current_stream(self.device))
                snapshot.record_stream(self.stream)
            scope = torch.cuda.stream(self.stream) if self.is_cuda else contextlib.nullcontext()
            with scope:
                if source_ready is not None:
                    self.stream.wait_event(source_ready)
                begin = torch.cuda.Event(enable_timing=True) if self.is_cuda else None
                if begin is not None:
                    begin.record()
                decoded = self.vae.model.cached_decode(snapshot.permute(0, 2, 1, 3, 4), self.scale).float().clamp_(-1, 1)
                expected = 4*latent.shape[1]-(3 if start_latent == 0 else 0)
                if decoded.shape[2] != expected:
                    raise RuntimeError('VAE temporal cache continuity or output frame count failed')
                shape = (1, decoded.shape[1], 4*self.max_chunk_latents, *decoded.shape[3:])
                if self.buffers is None:
                    required = self.slots*math.prod(shape)*decoded.element_size()
                    if required > self.budget:
                        raise ValueError(f'VAE output pool requires {required} bytes, budget={self.budget}')
                    self.buffers = [torch.empty(shape, dtype=decoded.dtype, device='cpu', pin_memory=self.is_cuda)
                                    for _ in range(self.slots)]
                if self.buffers[slot].shape != shape:
                    raise ValueError('VAE output geometry changed within a stream')
                # A short first chunk must still have a contiguous host DMA
                # destination, rather than a channel-strided [:,:,:9] view of12.
                view = self.buffers[slot].view(-1)[:decoded.numel()].view(decoded.shape)
                view.copy_(decoded, non_blocking=self.is_cuda)
                ready = torch.cuda.Event(enable_timing=True) if self.is_cuda else None
                if ready is not None:
                    ready.record()
            record = dict(start_latent=int(start_latent), latent_frames=int(latent.shape[1]),
                          start_pixel=self.pixels_submitted, pixel_frames=expected,
                          submitted_s=time.perf_counter()-self.started)
            self.latents_submitted += latent.shape[1]
            self.pixels_submitted += expected
            self.jobs.put((slot, view, ready, begin, record, snapshot, decoded))
        except BaseException:
            if self.stream is not None:
                self.stream.synchronize()
            self.free.put(slot)
            raise

    def finish(self):
        if self.closed:
            raise RuntimeError('streaming VAE already finished')
        self.closed = True
        self.jobs.put(None)
        self.worker.join(timeout=60)
        if self.worker.is_alive():
            raise RuntimeError('VAE completion worker did not drain')
        if self.stream is not None:
            self.stream.synchronize()
        self.vae.model.clear_cache()
        if self.error is not None:
            raise RuntimeError('streaming VAE completion failed') from self.error
        if sum(r['pixel_frames'] for r in self.records) != 4*self.latents_submitted-3:
            raise RuntimeError('stream ended with missing pixels')
        video = torch.cat(self.outputs, dim=1) if self.collect else None
        return video, dict(completed_latents=self.latents_submitted, completed_pixels=self.pixels_submitted,
            pinned_output_bytes=sum(x.numel()*x.element_size() for x in self.buffers) if self.is_cuda and self.buffers else 0,
            pinned_output_budget_bytes=self.budget, slots=self.slots,
            first_pixels_ready_s=self.records[0]['pixels_ready_s'],
            first_sink_completed_s=self.records[0]['sink_completed_s'],
            complete_s=time.perf_counter()-self.started, producer_backpressure_s=self.producer_backpressure_s,
            records=self.records, collected_CPU_outputs_unbounded_by_slot_budget=self.collect,
            measured_GPU_overlap_requires_Nsight=True)
