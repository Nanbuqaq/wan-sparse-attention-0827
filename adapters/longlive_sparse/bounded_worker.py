"""Single-consumer slot ownership with failure-aware producer backpressure."""
from queue import Empty,Full,Queue
import threading
import time


class BoundedWorker:
    def __init__(self,slots,process,*,name='bounded-worker'):
        if slots<1:raise ValueError('positive slot count required')
        self.free=Queue(maxsize=slots);self.jobs=Queue(maxsize=slots)
        for i in range(slots):self.free.put(i)
        self.process=process;self.error=None;self.closed=False;self.stop=threading.Event()
        self.backpressure_s=0.;self.completed=0
        self.thread=threading.Thread(target=self._run,name=name,daemon=True);self.thread.start()

    def check(self):
        if self.error is not None:raise RuntimeError('pipeline consumer failed') from self.error
        if self.closed or self.stop.is_set():raise RuntimeError('pipeline is closed')

    def reserve(self):
        began=time.perf_counter()
        while True:
            self.check()
            try:slot=self.free.get(timeout=.1);break
            except Empty:pass
        self.backpressure_s+=time.perf_counter()-began
        return slot

    def release_unsubmitted(self,slot):self.free.put_nowait(slot)

    def enqueue(self,slot,item):
        self.check()
        self.jobs.put_nowait((slot,item))

    def wait_completed(self,count):
        while self.completed<count:
            self.check();self.stop.wait(.01)

    def _run(self):
        try:
            while not self.stop.is_set():
                try:job=self.jobs.get(timeout=.1)
                except Empty:continue
                if job is None:return
                slot,item=job
                try:self.process(slot,item);self.completed+=1
                finally:self.free.put_nowait(slot)
        except BaseException as error:
            self.error=error;self.stop.set()

    def finish(self,timeout=60):
        self.check()
        while True:
            self.check()
            try:self.jobs.put(None,timeout=.1);break
            except Full:pass
        self.closed=True;self.thread.join(timeout=timeout)
        if self.thread.is_alive():raise RuntimeError('pipeline consumer did not drain')
        if self.error is not None:raise RuntimeError('pipeline consumer failed') from self.error

    def abort(self,timeout=60):
        self.closed=True;self.stop.set();self.thread.join(timeout=timeout)
        if self.thread.is_alive():raise RuntimeError('pipeline consumer did not stop')
