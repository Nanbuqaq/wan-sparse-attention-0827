import torch

from adapters.longlive_sparse.startup_host_trace import StartupHostTrace


def test_nested_exclusive_times_do_not_double_count():
    trace=StartupHostTrace()
    with trace.phase('parent'):
        with trace.phase('child'):sum(range(100))
    parent,child=trace.events
    assert parent['duration_ns']==parent['exclusive_host_ns']+child['duration_ns']
    assert child['parent']==parent['id'] and child['exclusive_host_ns']>=0
    assert len(trace.chrome()['traceEvents'])==2


def test_wrapped_initialization_preserves_tensor_and_RNG_values():
    trace=StartupHostTrace();torch.manual_seed(7)
    expected=torch.nn.init.normal_(torch.empty(20));expected_next=torch.rand(3)
    torch.manual_seed(7)
    actual=trace.wrap('normal',torch.nn.init.normal_)(torch.empty(20));actual_next=torch.rand(3)
    assert torch.equal(expected,actual) and torch.equal(expected_next,actual_next)
    assert trace.aggregate()['normal']['calls']==1


def test_failed_scope_keeps_record_and_unwinds_stack():
    trace=StartupHostTrace()
    try:
        with trace.phase('fails'):raise ValueError('test')
    except ValueError:pass
    assert trace.events[0]['status']=='failed' and not any(trace.stacks.values())
