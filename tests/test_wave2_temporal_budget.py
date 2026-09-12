import torch
import pytest
from pathlib import Path
from adapters.longlive_sparse.wave2_temporal_budget import updated_owners,classify_window
from scripts.run_longlive2_native_reference import native_cut_schedule


def test_storage_version_survives_roll_and_current_write_gets_new_version():
    old=[('native',i,1,0.) for i in range(8)]
    old[2]=('recalled',40,2,'binding',8.,88,24.)
    info=dict(action='roll_and_insert',sink_tokens=2,num_evicted_tokens=2,num_rolled_tokens=4,
        local_start_index=6,local_end_index=8,pinned_shift=2)
    owners=updated_owners(old,info,8,1,9,24.)
    assert owners[:2]==old[:2] and owners[2:6]==old[4:8]
    assert owners[6:]==[('native',8,9,24.),('native',9,9,24.)]
    assert old[2][0]=='recalled'


def test_protected_union_uses_real_slots_and_recall_content_not_frame_id_only():
    owners=[('native',i,1,0.) for i in range(8)]
    owners[4]=('recalled',0,2,'newbinding',8.,88,24.)
    info=dict(local_start_index=6,local_end_index=8,pinned_shift=2)
    roles=classify_window(owners,list(range(8)),info,1,2,2,6,1)
    assert roles[4]['pin'] and roles[4]['recalled']
    assert sum(any(x.values()) for x in roles)==5
    owners[3]=None
    with pytest.raises(RuntimeError):classify_window(owners,list(range(8)),info,1,2,2,6,1)


def test_wave2_continuous_and_settled_schedule_boundaries():
    root=Path(__file__).resolve().parents[1]
    _,continuous=native_cut_schedule(root,'w2_rotating_wooden_bird')
    assert len(continuous[0])==16 and not any(p.startswith('The scene transitions.') for p in continuous[0])
    _,state=native_cut_schedule(root,'w2_settled_pebble_bowl')
    assert [i for i,p in enumerate(state[0]) if p.startswith('The scene transitions.')]==[6,12]
    _,gate=native_cut_schedule(root,'w2_settled_pebble_bowl',gate=True,episode_gate=True)
    assert len(gate[0])==8 and [i for i,p in enumerate(gate[0]) if p.startswith('The scene transitions.')]==[2,6]


def test_frozen_wave_is_twelve_cases_without_fake_continuous_recall_arms(tmp_path):
    import json
    from scripts.run_native_duration_wave import build_wave2_cases
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())
    valid=[s['id'] for s in spec['scenarios']]
    native=build_wave2_cases(spec,'native',tmp_path,tmp_path,tmp_path,20261010)
    algorithms=build_wave2_cases(spec,'algorithms',tmp_path,tmp_path,tmp_path,20261010,valid)
    assert len(native)==4 and len(algorithms)==8 and len({c['id'] for c in native+algorithms})==12
    for c in native+algorithms:
        assert '--native-inplace-cache' in c['cmd'] and '--native-shared-conditioning' in c['cmd']
        assert '--resident-history-policy' not in c['cmd'] and '--causal-scene-memory' not in c['cmd']
    assert all('--wave2-capture' not in c['cmd'] for c in native)
    with pytest.raises(ValueError):build_wave2_cases(spec,'algorithms',tmp_path,tmp_path,tmp_path,20261010)
