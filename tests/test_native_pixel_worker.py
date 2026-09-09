import contextlib
import threading
import time

import pytest
import torch

from adapters.longlive_sparse.bounded_worker import BoundedWorker
from adapters.longlive_sparse.native_video_pipeline import NativeVideoPipeline,pinned_pool_bytes


def fake_pipeline(sink):
    pipe=object.__new__(NativeVideoPipeline);pipe.encoded_pixels=0;pipe.sink=sink
    pipe.span=lambda *args,**kwargs:contextlib.nullcontext();pipe._time=time.perf_counter
    return pipe


def test_native_double_pixel_pool_stays_within_128MiB():
    shape=(1,128,48,44,80)
    assert pinned_pool_bytes(shape,pixel_slots=1)==(5406720,43253760)
    a,b=pinned_pool_bytes(shape,pixel_slots=2)
    assert (a,b)==(5406720,86507520) and a+b<128*1024**2
    with pytest.raises(ValueError):pinned_pool_bytes(shape,pixel_slots=0)


def test_sink_completion_and_chunk_delivery_are_marked_after_consume():
    observed=[];pipe=fake_pipeline(lambda x:observed.append(x.clone()))
    host=torch.arange(12.).reshape(1,3,4,1,1);row=dict(start_pixel=0,pixel_frames=4);record={}
    pipe._encode(0,(host,row,record,True))
    assert torch.equal(observed[0],host.permute(0,2,1,3,4))
    assert pipe.encoded_pixels==4 and record['finished_s']==row['sink_finished_s']
    with pytest.raises(RuntimeError):pipe._encode(0,(host,row,{},True))


def test_slow_encoder_retains_slots_and_preserves_every_group():
    entered=threading.Event();release=threading.Event();seen=[]
    def sink(x):
        entered.set();assert release.wait(timeout=2);seen.append(float(x.flatten()[0]))
    pipe=fake_pipeline(sink);worker=BoundedWorker(1,pipe._encode);buffer=torch.zeros(1,3,4,1,1)
    slot=worker.reserve();buffer.fill_(1);row=dict(start_pixel=0,pixel_frames=4)
    worker.enqueue(slot,(buffer,row,{},True));assert entered.wait(timeout=2)
    assert worker.free.empty()
    release.set();worker.wait_completed(1)
    for i in range(2,10):
        slot=worker.reserve();buffer.fill_(i)
        worker.enqueue(slot,(buffer,dict(start_pixel=(i-1)*4,pixel_frames=4),{},True))
    worker.finish();assert seen==list(map(float,range(1,10))) and pipe.encoded_pixels==36


def test_encoder_error_is_visible_to_pipeline_producer():
    def bad(x):raise ValueError('codec failed')
    pipe=fake_pipeline(bad);pipe.encoder_worker=BoundedWorker(1,pipe._encode)
    pipe.worker=type('Producer',(),{'check':lambda self:None})()
    slot=pipe.encoder_worker.reserve()
    pipe.encoder_worker.enqueue(slot,(torch.zeros(1,3,1,1,1),dict(start_pixel=0,pixel_frames=1),{},True))
    pipe.encoder_worker.thread.join(timeout=2)
    with pytest.raises(RuntimeError):pipe._check_workers()
    pipe.encoder_worker.abort()
