import threading
import fcntl

import pytest

from adapters.longlive_sparse.bounded_worker import BoundedWorker
from scripts.run_on_free_gpu import locked_gpu_set


def test_slots_are_returned_only_after_consumer_and_order_is_preserved():
    release=threading.Event();started=threading.Event();seen=[]
    def process(slot,value):
        started.set();assert release.wait(2);seen.append(value)
    worker=BoundedWorker(1,process);slot=worker.reserve();worker.enqueue(slot,1)
    assert started.wait(2) and worker.free.empty()
    release.set();worker.wait_completed(1)
    slot=worker.reserve();worker.enqueue(slot,2);worker.finish()
    assert seen==[1,2] and worker.free.qsize()==1


def test_consumer_failure_is_not_an_infinite_producer_wait():
    def process(slot,value):raise ValueError('injected sink failure')
    worker=BoundedWorker(1,process);worker.enqueue(worker.reserve(),None);worker.thread.join(2)
    with pytest.raises(RuntimeError,match='consumer failed'):worker.reserve()
    with pytest.raises(RuntimeError,match='consumer failed'):worker.finish()
    worker.abort()


def test_multi_gpu_lock_is_atomic_and_releases_on_partial_failure(tmp_path):
    rows=[dict(index=i,memory=2,utilization=0) for i in (0,1)]
    with locked_gpu_set([0,1],rows,lock_dir=tmp_path):
        with pytest.raises(RuntimeError,match='lock is busy'):
            with locked_gpu_set([1,0],rows,lock_dir=tmp_path):pass
    with locked_gpu_set([1,0],rows,lock_dir=tmp_path):pass
    with pytest.raises(RuntimeError,match='not all'):
        with locked_gpu_set([0,1],[rows[0],dict(index=1,memory=2000,utilization=80)],lock_dir=tmp_path):pass


def test_duplicate_physical_device_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        with locked_gpu_set([0,0],[],lock_dir=tmp_path):pass


def test_partial_gpu_set_acquisition_does_not_leak_first_lock(tmp_path):
    rows=[dict(index=i,memory=2,utilization=0) for i in (0,1)]
    with (tmp_path/'wan_sparse_gpu_1.lock').open('w') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match='lock is busy'):
            with locked_gpu_set([0,1],rows,lock_dir=tmp_path):pass
        with (tmp_path/'wan_sparse_gpu_0.lock').open('w') as available:
            fcntl.flock(available,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(available,fcntl.LOCK_UN)
