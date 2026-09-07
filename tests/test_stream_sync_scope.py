from types import SimpleNamespace
import threading
import pytest

from adapters.longlive_sparse.profiling import cuda_sync_scope, synchronize_cuda, module_cuda_sync_scope


def test_stream_scope_is_explicit_nested_and_thread_local(monkeypatch):
    import torch
    calls = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device=None: calls.append(('device', device)))
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device=None:
        SimpleNamespace(synchronize=lambda: calls.append(('stream', device))))
    synchronize_cuda(0)
    with cuda_sync_scope('current_stream'):
        synchronize_cuda(0)
        thread = threading.Thread(target=lambda: synchronize_cuda(1))
        thread.start(); thread.join()
        with pytest.raises(RuntimeError):
            with cuda_sync_scope('device'):
                synchronize_cuda(0)
                raise RuntimeError('restore outer scope')
        synchronize_cuda(0)
    synchronize_cuda(0)
    assert calls == [('device', 0), ('stream', 0), ('device', 1), ('device', 0), ('stream', 0), ('device', 0)]


def test_module_scope_does_not_change_public_signature(monkeypatch):
    import inspect
    import torch
    calls = []
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device=None:
        SimpleNamespace(synchronize=lambda: calls.append(device)))
    @module_cuda_sync_scope
    def forward(self, query, *, named=None):
        synchronize_cuda(named)
        return query
    owner = SimpleNamespace(system_config=SimpleNamespace(cuda_sync_scope='current_stream'))
    assert forward(owner, 8, named=3) == 8 and calls == [3]
    assert list(inspect.signature(forward).parameters) == ['self', 'query', 'named']
