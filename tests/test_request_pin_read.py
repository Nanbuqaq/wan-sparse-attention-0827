import pytest
from adapters.longlive_sparse.request_pin_read import revision_positions


def example():
    frames=list(range(16))+list(range(24,40))
    owners=[('native',i,0,8.) for i in frames]
    roles=[dict(pin=8<=i<16,current=i>=24) for i in range(32)]
    return owners,list(range(32)),roles


def test_revision_separates_old_pin_and_equal_budget_recent_without_global_or_current():
    owners,physical,roles=example()
    a,da=revision_positions('drop_pin',owners,physical,roles,24,8)
    b,db=revision_positions('drop_recent',owners,physical,roles,24,8)
    assert da==list(range(8,16)) and db==list(range(16,24))
    assert len(a)==len(b)==24
    assert all(i in a and i in b for i in list(range(8))+list(range(24,32)))


def test_current_request_pin_is_not_removed_and_insufficient_control_budget_fails():
    owners,physical,roles=example()
    assert revision_positions('drop_pin',owners,physical,roles,8,8)[1]==[]
    with pytest.raises(RuntimeError):revision_positions('drop_recent',owners,physical[:16],roles[:16],24,8)
