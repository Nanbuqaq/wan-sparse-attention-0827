import threading
import time

import pytest

from adapters.longlive_sparse.bounded_video_writer import BoundedVideoWriter


def test_seek_order_and_partial_writes_preserve_bytes(tmp_path,monkeypatch):
    path=tmp_path/'output.bin'
    original=BoundedVideoWriter._open_file
    class Partial:
        def __init__(self,file):self.file=file
        def __getattr__(self,name):return getattr(self.file,name)
        def write(self,data):return self.file.write(data[:3])
    monkeypatch.setattr(BoundedVideoWriter,'_open_file',lambda self:Partial(original(self)))
    writer=BoundedVideoWriter(path,buffer_bytes=16,slot_bytes=8,started=time.perf_counter())
    writer.write(b'abcdefghijklmnopqrst')
    assert writer.seek(2)==2
    writer.write(b'XYZ')
    assert writer.seek(-2,2)==18
    writer.write(b'!!')
    result=writer.close()
    assert path.read_bytes()==b'abXYZfghijklmnopqr!!'
    assert result['accepted_write_bytes']==result['completed_write_bytes']==25
    assert result['peak_pending_payload_bytes']<=16
    assert result['file_bytes']==20 and result['status']=='drained'


def test_inflight_slot_is_charged_and_producer_waits(tmp_path,monkeypatch):
    entered=threading.Event();release=threading.Event();completed=threading.Event();errors=[]
    original=BoundedVideoWriter._open_file
    class Slow:
        def __init__(self,file):self.file=file
        def __getattr__(self,name):return getattr(self.file,name)
        def write(self,data):
            entered.set()
            if not release.wait(3):raise RuntimeError('test release missing')
            return self.file.write(data)
    monkeypatch.setattr(BoundedVideoWriter,'_open_file',lambda self:Slow(original(self)))
    path=tmp_path/'slow.bin'
    writer=BoundedVideoWriter(path,buffer_bytes=16,slot_bytes=8,started=time.perf_counter())
    def produce():
        try:writer.write(b'x'*24)
        except BaseException as error:errors.append(error)
        finally:completed.set()
    thread=threading.Thread(target=produce);thread.start()
    try:
        assert entered.wait(1)
        assert not completed.wait(.1)
        assert writer.pending_bytes<=16 and writer.peak_pending_bytes<=16
    finally:release.set();thread.join(3)
    assert completed.is_set() and not errors
    result=writer.close()
    assert path.read_bytes()==b'x'*24 and result['producer_buffer_wait_s']>=.09


def test_writer_failure_wakes_producer_and_fails_delivery(tmp_path,monkeypatch):
    def fail(self):raise OSError('simulated storage failure')
    monkeypatch.setattr(BoundedVideoWriter,'_open_file',fail)
    writer=BoundedVideoWriter(tmp_path/'fail.bin',buffer_bytes=8,slot_bytes=8,started=time.perf_counter())
    with pytest.raises(RuntimeError,match='consumer failed'):writer.write(b'x'*16)
    with pytest.raises(RuntimeError,match='consumer failed'):writer.close()
    writer.abort()


def test_existing_output_is_never_replaced(tmp_path):
    path=tmp_path/'existing.bin';path.write_bytes(b'original')
    writer=BoundedVideoWriter(path,buffer_bytes=8,slot_bytes=8,started=time.perf_counter())
    writer.write(b'changed')
    with pytest.raises(RuntimeError):writer.close()
    assert path.read_bytes()==b'original'
    writer.abort()


def test_close_charges_wait_until_output_is_delivered(tmp_path,monkeypatch):
    entered=threading.Event();release=threading.Event();closed=threading.Event();results=[]
    original=BoundedVideoWriter._open_file
    class Slow:
        def __init__(self,file):self.file=file
        def __getattr__(self,name):return getattr(self.file,name)
        def write(self,data):
            entered.set()
            if not release.wait(3):raise RuntimeError('test release missing')
            return self.file.write(data)
    monkeypatch.setattr(BoundedVideoWriter,'_open_file',lambda self:Slow(original(self)))
    path=tmp_path/'drain.bin'
    writer=BoundedVideoWriter(path,buffer_bytes=16,slot_bytes=8,started=time.perf_counter())
    writer.write(b'abcdefghijklmnop')
    def finish():
        try:results.append(writer.close())
        finally:closed.set()
    thread=threading.Thread(target=finish);thread.start()
    try:
        assert entered.wait(1)
        assert not closed.wait(.1)
    finally:release.set();thread.join(3)
    assert len(results)==1 and results[0]['drain_and_close_s']>=.09
    assert path.read_bytes()==b'abcdefghijklmnop'
