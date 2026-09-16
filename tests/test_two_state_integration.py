"""Integration check: two_state VersionSceneMemory saves closed+open and freezes."""
import torch
from adapters.longlive_sparse.version_two_state_bank import TwoStateBank


def _win(values_k, values_v, ft=1):
    # a fake single-layer cache holding one 8-frame window
    return dict(k=torch.tensor(values_k, dtype=torch.bfloat16).reshape(1, 8 * ft, 1, 1),
                v=torch.tensor(values_v, dtype=torch.bfloat16).reshape(1, 8 * ft, 1, 1),
                local_end_index=8 * ft, global_end_index=8 * ft)


def test_two_state_bank_freeze_requires_same_object():
    bank = TwoStateBank(1, 64)
    closed = [dict(frame=f, phase=0.) for f in range(8, 16)]
    open_ = [dict(frame=f, phase=0.) for f in range(24, 32)]
    bank.save_state([_win([8,9,10,11,12,13,14,15],[0]*8)], closed, state='old')
    bank.save_state([_win([24,25,26,27,28,29,30,31],[0]*8)], open_, state='new')
    # closed prompt vs open prompt share the red-toolbox descriptor -> high cosine
    bank.freeze(same_object_cosine=0.9)
    assert [r['frame'] for r in bank.records] == [12, 13, 14, 15, 28, 29, 30, 31]
    k, v = bank.kv()[0]
    assert k.shape[1] == 8 and v.shape[1] == 8
    # old4 = closed trailing (12-15), new4 = open trailing (28-31)
    assert [r['version'] for r in bank.records] == [1, 1, 1, 1, 2, 2, 2, 2]
