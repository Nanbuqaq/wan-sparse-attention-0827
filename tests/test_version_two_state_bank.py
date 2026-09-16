import torch
from adapters.longlive_sparse.version_two_state_bank import TwoStateBank


def _cache(frames_k, frames_v):
    return dict(k=torch.tensor(frames_k, dtype=torch.bfloat16).reshape(1, 8, 1, 1),
                v=torch.tensor(frames_v, dtype=torch.bfloat16).reshape(1, 8, 1, 1),
                local_end_index=8)


def test_two_state_save_pre_post_update_then_freeze():
    # red toolbox: closed window 8-15 saved BEFORE the open update; open window 24-31 AFTER.
    closed_records = [dict(frame=f, phase=0.) for f in range(8, 16)]
    open_records = [dict(frame=f, phase=0.) for f in range(24, 32)]
    closed_cache = _cache([8, 9, 10, 11, 12, 13, 14, 15], [108, 109, 110, 111, 112, 113, 114, 115])
    open_cache = _cache([24, 25, 26, 27, 28, 29, 30, 31], [124, 125, 126, 127, 128, 129, 130, 131])
    bank = TwoStateBank(1, 64)
    bank.save_state([closed_cache], closed_records, state='old')   # pre-update
    bank.save_state([open_cache], open_records, state='new')       # post-update
    bank.freeze(same_object_cosine=0.93)
    assert bank.frozen and bank.raw_bytes == 32 and bank.D2H_bytes == 32
    # old4 = trailing four of closed (12-15), new4 = trailing four of open (28-31)
    assert [r['frame'] for r in bank.records] == [12, 13, 14, 15, 28, 29, 30, 31]
    assert [r['version'] for r in bank.records] == [1] * 4 + [2] * 4
    k, v = bank.kv()[0]
    assert k[0, :, 0, 0].tolist() == [12, 13, 14, 15, 28, 29, 30, 31]
    assert v[0, :, 0, 0].tolist() == [112, 113, 114, 115, 128, 129, 130, 131]
    assert bank.audit()['unique_owned_raw_bytes'] == 32


def test_freeze_requires_both_states_saved():
    bank = TwoStateBank(1, 64)
    closed_records = [dict(frame=f, phase=0.) for f in range(8, 16)]
    closed_cache = _cache([8, 9, 10, 11, 12, 13, 14, 15], [0] * 8)
    bank.save_state([closed_cache], closed_records, state='old')
    try:
        bank.freeze(same_object_cosine=0.95)
        raise SystemExit('freeze must require both old and new states')
    except RuntimeError as e:
        assert 'both old and new' in str(e)


def test_refuses_dissimilar_states_without_relaxing_threshold():
    closed_records = [dict(frame=f, phase=0.) for f in range(8, 16)]
    open_records = [dict(frame=f, phase=0.) for f in range(24, 32)]
    closed_cache = _cache([8, 9, 10, 11, 12, 13, 14, 15], [0] * 8)
    open_cache = _cache([24, 25, 26, 27, 28, 29, 30, 31], [0] * 8)
    bank = TwoStateBank(1, 64)
    bank.save_state([closed_cache], closed_records, state='old')
    bank.save_state([open_cache], open_records, state='new')
    try:
        bank.freeze(same_object_cosine=0.5)
        raise SystemExit('must refuse dissimilar states instead of forcing versions')
    except RuntimeError as e:
        assert 'same object' in str(e)


def test_staging_charged_and_double_save_rejected():
    closed_records = [dict(frame=f, phase=0.) for f in range(8, 16)]
    open_records = [dict(frame=f, phase=0.) for f in range(24, 32)]
    closed_cache = _cache([8, 9, 10, 11, 12, 13, 14, 15], [0] * 8)
    open_cache = _cache([24, 25, 26, 27, 28, 29, 30, 31], [0] * 8)
    # budget too small for BOTH halves (each half = 4 frames * 2 tensors * 2B = 16B; two = 32B)
    tight = TwoStateBank(1, 16)
    tight.save_state([closed_cache], closed_records, state='old')
    try:
        tight.save_state([open_cache], open_records, state='new')
        raise SystemExit('staging must be charged against the eight-frame budget')
    except RuntimeError as e:
        assert 'budget' in str(e)
    bank = TwoStateBank(1, 64)
    bank.save_state([closed_cache], closed_records, state='old')
    try:
        bank.save_state([closed_cache], closed_records, state='old')
        raise SystemExit('re-saving the same state must be rejected')
    except RuntimeError:
        pass


def test_non_chronological_states_rejected_at_freeze():
    # 'old' window later than 'new' window must not freeze
    later_records = [dict(frame=f, phase=0.) for f in range(24, 32)]
    earlier_records = [dict(frame=f, phase=0.) for f in range(8, 16)]
    later_cache = _cache([24, 25, 26, 27, 28, 29, 30, 31], [0] * 8)
    earlier_cache = _cache([8, 9, 10, 11, 12, 13, 14, 15], [0] * 8)
    bank = TwoStateBank(1, 64)
    bank.save_state([later_cache], later_records, state='old')
    bank.save_state([earlier_cache], earlier_records, state='new')
    try:
        bank.freeze(same_object_cosine=0.95)
        raise SystemExit('old must be strictly earlier than new')
    except ValueError:
        pass
