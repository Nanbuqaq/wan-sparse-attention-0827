from concurrent.futures import ThreadPoolExecutor
import pytest
from adapters.longlive_sparse.resident_output_experiment import schedule,FileBarrier

def test_schedule_balances_order_and_keeps_warmups_out():
    s=schedule(4)
    assert [r['mode'] for r in s[:2]]==['inline','thread']
    assert not any(r['measured'] for r in s[:2])
    assert [r['mode'] for r in s[2:]]==['inline','thread','thread','inline','inline','thread','thread','inline']
    with pytest.raises(ValueError):schedule(2)

def test_all_participants_share_future_epoch(tmp_path):
    bs=[FileBarrier(tmp_path,i,lanes=4,timeout=2) for i in range(4)]
    with ThreadPoolExecutor(4) as ex:rows=list(ex.map(lambda b:b.wait('round0'),bs))
    assert len({r['target_ns'] for r in rows})==1
    assert all(r['released_ns']>=r['target_ns'] for r in rows)
    with pytest.raises(RuntimeError,match='stale'):bs[0].wait('round0')

def test_peer_failure_and_missing_peer_are_not_silent(tmp_path):
    b=FileBarrier(tmp_path,0,lanes=2,timeout=.05)
    with pytest.raises(TimeoutError):b.wait('missing')
    b.fail('model failed')
    with pytest.raises(RuntimeError,match='peer'):b.wait('next')
