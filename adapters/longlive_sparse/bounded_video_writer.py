"""Seekable, single-producer output with a fixed owned compressed-byte pool."""
import heapq
import os
from pathlib import Path
import threading
import time

from .bounded_worker import BoundedWorker


class BoundedVideoWriter:
    def __init__(self,path,*,buffer_bytes,started,slot_bytes=64*1024,drain_timeout=180):
        if buffer_bytes<slot_bytes or slot_bytes<1 or buffer_bytes%slot_bytes:
            raise ValueError('output byte budget must contain whole positive slots')
        self.path=Path(path);self.name=str(self.path);self.started=started
        self.slot_bytes=slot_bytes;self.buffer_bytes=buffer_bytes;self.drain_timeout=drain_timeout
        self.pool=[bytearray(slot_bytes) for _ in range(buffer_bytes//slot_bytes)]
        self.position=0;self.extent=0;self.closed=False;self.close_error=None;self.file=None
        self.submitted=0;self.enqueued_bytes=0;self.written_bytes=0
        self.max_input_write_bytes=0
        self.pending_bytes=0;self.peak_pending_bytes=0;self.stats_lock=threading.Lock()
        self.first_write_completed_s=None;self.io_host_s=0.;self.longest_io=[]
        self.worker=BoundedWorker(len(self.pool),self._consume,name='compressed-video-file-writer')

    def writable(self):return True
    def readable(self):return False
    def seekable(self):return True
    def tell(self):return self.position

    def seek(self,offset,whence=os.SEEK_SET):
        if self.closed:raise ValueError('seek on closed output')
        self.worker.check()
        if whence==os.SEEK_SET:position=offset
        elif whence==os.SEEK_CUR:position=self.position+offset
        elif whence==os.SEEK_END:position=self.extent+offset
        else:raise ValueError('unsupported output seek mode')
        if position<0:raise ValueError('negative output seek')
        self.position=position
        return position

    def write(self,data):
        if self.closed:raise ValueError('write on closed output')
        view=memoryview(data).cast('B');total=len(view)
        self.max_input_write_bytes=max(self.max_input_write_bytes,total)
        for start in range(0,total,self.slot_bytes):
            n=min(self.slot_bytes,total-start);slot=self.worker.reserve()
            charged=False
            try:
                memoryview(self.pool[slot])[:n]=view[start:start+n]
                with self.stats_lock:
                    self.pending_bytes+=n;self.peak_pending_bytes=max(self.peak_pending_bytes,self.pending_bytes)
                    charged=True
                self.worker.enqueue(slot,(self.position,n))
            except BaseException:
                if charged:
                    with self.stats_lock:self.pending_bytes-=n
                self.worker.release_unsubmitted(slot)
                raise
            self.position+=n;self.extent=max(self.extent,self.position)
            self.submitted+=1;self.enqueued_bytes+=n
        return total

    def _open_file(self):return self.path.open('xb',buffering=0)

    def _consume(self,slot,job):
        offset,n=job;began=time.perf_counter()
        try:
            if self.file is None:self.file=self._open_file()
            if self.file.tell()!=offset:self.file.seek(offset)
            view=memoryview(self.pool[slot])[:n]
            while view:
                written=self.file.write(view)
                if written is None or written<=0:raise OSError('output write made no progress')
                view=view[written:]
            self.written_bytes+=n
            ended=time.perf_counter();elapsed=ended-began;self.io_host_s+=elapsed
            if self.first_write_completed_s is None:self.first_write_completed_s=ended-self.started
            item=(elapsed,began-self.started,n)
            if len(self.longest_io)<16:heapq.heappush(self.longest_io,item)
            elif item>self.longest_io[0]:heapq.heapreplace(self.longest_io,item)
        finally:
            with self.stats_lock:self.pending_bytes-=n

    def flush(self):
        if self.closed:raise ValueError('flush on closed output')
        self.worker.wait_completed(self.submitted)
        if self.file is not None:self.file.flush()

    def close(self):
        if self.closed:
            if self.close_error is not None:raise RuntimeError('output close previously failed') from self.close_error
            return self.result
        self.closed=True;began=time.perf_counter()
        try:
            self.worker.finish(timeout=self.drain_timeout)
            if self.written_bytes!=self.enqueued_bytes or self.pending_bytes:
                raise RuntimeError('output queue did not deliver all bytes')
            size=0 if self.file is None else os.fstat(self.file.fileno()).st_size
            if size!=self.extent:raise RuntimeError('output file extent differs from accepted writes')
            if self.file is not None:self.file.close()
        except BaseException as error:
            self.close_error=error
            if not self.worker.thread.is_alive() and self.file is not None:self.file.close()
            raise
        self.result=dict(status='drained',buffer_payload_bytes=self.buffer_bytes,slot_bytes=self.slot_bytes,
            slot_count=len(self.pool),maximum_AVIO_input_bytes=self.max_input_write_bytes,
            peak_pending_payload_bytes=self.peak_pending_bytes,accepted_write_bytes=self.enqueued_bytes,
            completed_write_bytes=self.written_bytes,file_bytes=size,write_jobs=self.submitted,
            producer_buffer_wait_s=self.worker.backpressure_s,drain_and_close_s=time.perf_counter()-began,
            file_io_host_s=self.io_host_s,first_file_write_completed_s=self.first_write_completed_s,
            longest_io_jobs=[dict(host_s=s,start_s=t,bytes=n) for s,t,n in sorted(self.longest_io,reverse=True)],
            drain_included_in_sink_close=True,fsync_added=False,client_display_latency_not_measured=True,
            memory_scope='fixed payload pool; Python queue metadata and caller-owned AVIO input are additional')
        return self.result

    def abort(self):
        self.closed=True
        if self.close_error is None:self.close_error=RuntimeError('output was aborted')
        self.worker.abort(timeout=self.drain_timeout)
        if self.file is not None:self.file.close()
