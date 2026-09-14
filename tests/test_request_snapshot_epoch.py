import pytest
from adapters.longlive_sparse.resident_snapshot import RequestWriteEpoch,oldest_same_phase_window


def test_native_transition_prefix_does_not_create_spurious_request_update():
    e=RequestWriteEpoch()
    e.observe(8.,8,'The scene transitions. The object stays open.')
    e.observe(8.,16,'The object stays open.')
    assert e.floor_for_closed_phase(8.)==8
    e.observe(8.,24,'The object is now closed.')
    assert e.floor_for_closed_phase(8.)==24
    e.observe(8.,40,'The object is now closed.')
    assert e.floor_for_closed_phase(8.)==24
    e.observe(16.,48,'A flower.')
    with pytest.raises(RuntimeError):e.floor_for_closed_phase(8.)
    with pytest.raises(ValueError):e.observe(16.,40,'A flower.')


def test_request_floor_excludes_obsolete_pin_without_fabricating_evicted_frames():
    owners=([('native',i,5,0.) for i in range(8)]
            +[('native',i,10,8.) for i in range(8,16)]
            +[('native',i,30,8.) for i in range(32,48)])
    assert oldest_same_phase_window(owners,8.)[0]==list(range(8,16))
    assert oldest_same_phase_window(owners,8.,minimum_frame=24)[0]==list(range(32,40))
    assert oldest_same_phase_window(owners,8.,minimum_frame=40)[0]==list(range(40,48))
    with pytest.raises(ValueError):oldest_same_phase_window(owners,8.,minimum_frame=44)
