"""Committed DiT chunks -> bounded host staging -> second GPU VAE -> sink.

CPU wall spans are exported as a Perfetto-readable trace, not asserted as GPU
overlap. NVTX labels permit a separate CUPTI/Nsys timeline audit after gates.
"""
from collections import Counter
import contextlib
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import torch

from .bounded_worker import BoundedWorker
from .native_vae_stream import NativeVAEStream
from .profiling import nvtx_range


def pinned_pool_bytes(latent_shape,*,slots=2,max_chunk=8,pixel_slots=1):
    if len(latent_shape)!=5 or latent_shape[0]!=1 or latent_shape[2]!=48 or min(latent_shape)<1:
        raise ValueError('positive batch1 native latent geometry required')
    if min(slots,max_chunk,pixel_slots)<1:raise ValueError('positive pool sizes required')
    input_bytes=slots*math.prod((1,max_chunk,*latent_shape[2:]))*2
    output_bytes=pixel_slots*4*3*(latent_shape[3]*16)*(latent_shape[4]*16)*4
    return input_bytes,output_bytes


class NativeVideoPipeline:
    def __init__(self,vae,unpatchify,sink,*,source_device,target_device,latent_shape,
                 started,slots=2,pinned_budget=128*1024**2,max_chunk=8,serial=False,
                 encode_mode='inline',pixel_slots=2):
        self.source=torch.device(source_device);self.target=torch.device(target_device)
        if self.source.type!='cuda' or self.target.type!='cuda' or self.source==self.target:
            raise ValueError('this qualified pipeline requires two distinct explicit CUDA devices')
        if len(latent_shape)!=5 or latent_shape[0]!=1 or latent_shape[2]!=48 or slots<1:
            raise ValueError('qualified batch1 native latent geometry required')
        if encode_mode not in ('inline','thread'):raise ValueError('unknown pixel encode mode')
        self.vae,self.sink=vae,sink;self.shape=tuple(latent_shape);self.started=started
        self.producer_tid=threading.get_native_id()
        self.slots=slots;self.max_chunk=max_chunk;self.frames=0;self.pixel_frames=0;self.submissions=0;self.serial=serial
        self.encode_mode=encode_mode;self.pixel_slots=1 if encode_mode=='inline' else pixel_slots
        self.encoded_pixels=0;self.group_submissions=0;self.encoder_worker=None
        self.counts=Counter();self.hook=None;self.records=[];self.trace=[];self.trace_lock=threading.Lock()
        self.dtype=torch.bfloat16;self.closed=False;self.budget=pinned_budget;self.active_decode_iterator=None
        latent_slot_shape=(1,max_chunk,*self.shape[2:])
        output_elements=4*3*(self.shape[3]*16)*(self.shape[4]*16)
        self.input_pinned_bytes,self.output_pinned_bytes=pinned_pool_bytes(self.shape,slots=slots,
            max_chunk=max_chunk,pixel_slots=self.pixel_slots)
        if self.input_pinned_bytes+self.output_pinned_bytes>pinned_budget:
            raise ValueError('requested pinned pools exceed explicit budget')
        self.inputs=[torch.empty(latent_slot_shape,dtype=self.dtype,pin_memory=True) for _ in range(slots)]
        self.pixel_buffers=[torch.empty(output_elements,dtype=torch.float32,pin_memory=True) for _ in range(self.pixel_slots)]
        self.pixel_buffer=self.pixel_buffers[0]
        self.d2h=torch.cuda.Stream(device=self.source)
        self.decode_stream=torch.cuda.Stream(device=self.target)
        with torch.cuda.device(self.target),torch.cuda.stream(self.decode_stream):
            scale=[vae.mean.to(device=self.target,dtype=self.dtype),1./vae.std.to(device=self.target,dtype=self.dtype)]
        self.decoder=NativeVAEStream(vae.model,scale,unpatchify)
        self.digest=hashlib.sha256();self.digest.update(str(self.dtype).encode());self.digest.update(json.dumps(list(self.shape)).encode())
        if encode_mode=='thread':self.encoder_worker=BoundedWorker(self.pixel_slots,self._encode,name='native-pixel-encode-worker')
        self.worker=BoundedWorker(slots,self._consume,name='native-vae-gpu-worker')

    def _time(self):return time.perf_counter()-self.started

    def _check_workers(self):
        self.worker.check()
        if self.encoder_worker is not None:self.encoder_worker.check()

    def _encode(self,slot,job):
        host,row,record,is_last=job
        if row['start_pixel']!=self.encoded_pixels:raise RuntimeError('pixel sink order changed')
        with self.span('encode',start_pixel=row['start_pixel']):self.sink(host.permute(0,2,1,3,4))
        self.encoded_pixels+=row['pixel_frames'];row['sink_finished_s']=self._time()
        if is_last:record['finished_s']=row['sink_finished_s']

    @contextlib.contextmanager
    def span(self,name,**args):
        begin=self._time()
        with nvtx_range('native_pipeline/'+name):
            try:yield
            finally:
                event=dict(name=name,ph='X',pid=1,tid=threading.get_native_id(),ts=begin*1e6,
                    dur=(self._time()-begin)*1e6,args=args)
                with self.trace_lock:self.trace.append(event)

    def attach(self,pipe):
        if pipe.sampling_steps!=4 or pipe.frame_seq_length<1:raise ValueError('qualified native four-step loop required')
        def committed(owner,values,kwargs,result):
            frame=int(kwargs['current_start'])//pipe.frame_seq_length;self.counts[frame]+=1
            self._check_workers()
            if self.counts[frame]!=pipe.sampling_steps+1:return
            if not bool((kwargs['timestep']==0).all()):raise RuntimeError('only clean committed chunks can be submitted')
            self.submit(kwargs['noisy_image_or_video'],start_latent=frame)
        self.hook=pipe.generator.register_forward_hook(committed,with_kwargs=True)

    def detach(self):
        if self.hook is not None:self.hook.remove();self.hook=None

    def submit(self,latent,*,start_latent):
        if self.closed:raise RuntimeError('pipeline is closed')
        self._check_workers()
        if latent.device!=self.source or latent.dtype!=self.dtype or tuple(latent.shape[2:])!=self.shape[2:]:
            raise ValueError('source latent device/dtype/geometry changed')
        if start_latent!=self.frames or not 0<latent.shape[1]<=self.max_chunk or self.frames+latent.shape[1]>self.shape[1]:
            raise ValueError('committed chunks must arrive in order without gaps')
        with self.span('producer_backpressure',start_latent=start_latent):slot=self.worker.reserve()
        try:
            with self.span('latent_snapshot_and_D2H_submit',start_latent=start_latent):
                owned=latent.detach().clone()
                source_ready=torch.cuda.Event();source_ready.record(torch.cuda.current_stream(self.source))
                view=self.inputs[slot].view(-1)[:owned.numel()].view(owned.shape)
                begin=torch.cuda.Event(enable_timing=True);ready=torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(self.d2h):
                    self.d2h.wait_event(source_ready);begin.record();view.copy_(owned,non_blocking=True);ready.record()
                owned.record_stream(self.d2h)
                record=dict(start_latent=start_latent,latent_frames=latent.shape[1],submitted_s=self._time(),
                    latent_D2H_bytes=owned.numel()*owned.element_size(),latent_H2D_bytes=owned.numel()*owned.element_size())
                self.worker.enqueue(slot,(view,owned,begin,ready,record));self.frames+=latent.shape[1];self.submissions+=1
        except BaseException:
            self.d2h.synchronize();self.worker.release_unsubmitted(slot);raise
        if self.serial:
            with self.span('serial_control_wait',start_latent=start_latent):
                self.worker.wait_completed(self.submissions)
                if self.encoder_worker is not None:self.encoder_worker.wait_completed(self.group_submissions)

    def _consume(self,slot,job):
        view,owned,copy_begin,source_ready,record=job
        with torch.cuda.device(self.target),torch.cuda.stream(self.decode_stream),torch.inference_mode():
            with self.span('wait_latent_D2H',start_latent=record['start_latent']):source_ready.synchronize()
            record['D2H_GPU_stream_span_ms']=copy_begin.elapsed_time(source_ready)
            self.digest.update(view.contiguous().view(torch.uint8).numpy().tobytes())
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            with self.span('latent_H2D',start_latent=record['start_latent']):
                begin.record();gpu_latent=view.to(self.target,non_blocking=True);end.record();end.synchronize()
            record['H2D_GPU_stream_span_ms']=begin.elapsed_time(end)
            # This input slot stays owned through decoding. Separate output
            # slots stay owned until their entire CPU sink call has returned.
            del owned
            record['decode_started_s']=self._time();first_pixel=self.pixel_frames;groups=[]
            iterator=self.decoder.iter_decode(gpu_latent.permute(0,2,1,3,4))
            self.active_decode_iterator=iterator
            while True:
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                try:
                    with self.span('VAE_decode_group',start_latent=record['start_latent']):
                        begin.record();decoded=next(iterator);end.record()
                except StopIteration:break
                raw=decoded.float().clamp_(-1,1).contiguous()
                pixels=raw.shape[2]
                if pixels!=(1 if self.pixel_frames==0 else 4):raise RuntimeError('native temporal output continuity failed')
                output_slot=None;enqueued=False
                try:
                    with self.span('pixel_output_backpressure',start_pixel=self.pixel_frames):
                        output_slot=self.encoder_worker.reserve() if self.encoder_worker is not None else 0
                    buffer_acquired_s=self._time()
                    host=self.pixel_buffers[output_slot][:raw.numel()].view(raw.shape)
                    copied=torch.cuda.Event(enable_timing=True);copy_started=torch.cuda.Event(enable_timing=True)
                    with self.span('pixels_D2H',start_pixel=self.pixel_frames):
                        copy_started.record();host.copy_(raw,non_blocking=True);copied.record();copied.synchronize()
                    row=dict(start_pixel=self.pixel_frames,pixel_frames=pixels,CPU_pixels_ready_s=self._time(),
                        decode_GPU_stream_span_ms=begin.elapsed_time(end),pixel_D2H_GPU_stream_span_ms=copy_started.elapsed_time(copied),
                        pixel_D2H_bytes=raw.numel()*raw.element_size(),pixel_buffer_slot=output_slot,
                        pixel_buffer_acquired_s=buffer_acquired_s)
                    job=(host,row,record,len(groups)+1==record['latent_frames'])
                    if self.encoder_worker is None:self._encode(output_slot,job)
                    else:self.encoder_worker.enqueue(output_slot,job);enqueued=True
                    self.group_submissions+=1;self.pixel_frames+=pixels;groups.append(row)
                except BaseException:
                    # Keep any DMA-owned buffer alive until the CUDA stream is drained.
                    self.decode_stream.synchronize()
                    if self.encoder_worker is not None and output_slot is not None and not enqueued:
                        self.encoder_worker.release_unsubmitted(output_slot)
                    raise
            self.active_decode_iterator=None
            record.update(pixel_start=first_pixel,pixel_frames=self.pixel_frames-first_pixel,
                decode_and_enqueue_finished_s=self._time(),groups=groups)
            if self.encoder_worker is None:record['finished_s']=self._time()
            expected=4*record['latent_frames']-(3 if record['start_latent']==0 else 0)
            if record['pixel_frames']!=expected:raise RuntimeError('chunk pixel count mismatch')
            self.records.append(record)

    def finish(self,*,generation_finished_s):
        self.detach();self.worker.finish()
        if self.encoder_worker is not None:self.encoder_worker.finish()
        self.closed=True;self.decode_stream.synchronize();self.decoder.finish()
        if self.frames!=self.shape[1] or self.pixel_frames!=4*self.shape[1]-3:raise RuntimeError('pipeline ended with missing frames')
        if self.encoded_pixels!=self.pixel_frames:raise RuntimeError('pixel encoding ended with missing frames')
        pixels=self.sink.close()
        with self.trace_lock:
            self.trace.append(dict(name='generation_host',ph='X',pid=1,tid=self.producer_tid,ts=0,dur=generation_finished_s*1e6,args={}))
        return pixels,dict(status='pass',generation_host_s=generation_finished_s,complete_s=self._time(),
            scheduling='serial_two_gpu' if self.serial else 'overlap_two_gpu',
            producer_backpressure_s=self.worker.backpressure_s,source_device=str(self.source),target_device=str(self.target),
            latent_shape=list(self.shape),streamed_latent_sha256=self.digest.hexdigest(),completed_pixel_frames=self.pixel_frames,
            latent_D2H_bytes=sum(r['latent_D2H_bytes'] for r in self.records),latent_H2D_bytes=sum(r['latent_H2D_bytes'] for r in self.records),
            pixel_D2H_bytes=sum(g['pixel_D2H_bytes'] for r in self.records for g in r['groups']),
            pinned_input_bytes=self.input_pinned_bytes,pinned_output_bytes=self.output_pinned_bytes,pinned_budget_bytes=self.budget,
            encode_mode=self.encode_mode,pixel_buffer_slots=self.pixel_slots,encoded_pixels=self.encoded_pixels,
            pixel_output_backpressure_s=self.encoder_worker.backpressure_s if self.encoder_worker is not None else 0.,
            slot_count=self.slots,records=self.records,CPU_spans_and_GPU_stream_spans_not_additive=True,
            actual_GPU_overlap_requires_CUPTI_timeline=True,client_display_latency_not_measured=True)

    def write_trace(self,path):
        with Path(path).open('x') as handle:json.dump(dict(traceEvents=self.trace,displayTimeUnit='ms',
            metadata=dict(scope='CPU host spans only; use matching NVTX/CUPTI for GPU overlap')),handle)

    def abort(self):
        self.detach()
        # Wake a VAE producer blocked on pixel slots before joining that producer.
        if self.encoder_worker is not None:self.encoder_worker.stop.set()
        self.worker.abort()
        if self.encoder_worker is not None:self.encoder_worker.abort()
        self.closed=True
        if self.active_decode_iterator is not None:
            with contextlib.suppress(Exception):self.active_decode_iterator.close()
            self.active_decode_iterator=None
        with contextlib.suppress(Exception):self.d2h.synchronize();self.decode_stream.synchronize();self.decoder.finish()
        if not self.sink.closed:
            with contextlib.suppress(Exception):self.sink.close()
