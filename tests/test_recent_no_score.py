import pytest
from adapters.longlive_sparse.wave2_temporal_budget import recent_positions


def test_actual_frame_recency_and_protected_context():
    eligible=[(3,('native',20,1,0.)),(6,('native',18,1,0.)),(4,('native',30,1,0.)),(5,('native',21,1,0.))]
    positions,budget=recent_positions(eligible,[0,1,7],880,.5)
    assert positions==[0,1,4,5,7] and budget==1760


def test_initial_and_full_optional_windows_keep_exact_half():
    for count in (8,16):
        eligible=[(8+i,('native',i+8,1,0.)) for i in range(count)]
        positions,budget=recent_positions(eligible,list(range(8)),880,.5)
        assert len(positions)==8+count//2 and budget==count*880//2
    with pytest.raises(ValueError):recent_positions([(1,('native',8,1,0.))],[],880,.5)


def test_registered_recent_cohort_has_own_native_and_no_observer(tmp_path):
    import json
    from pathlib import Path
    from scripts.run_native_duration_wave import build_wave2_cases
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'recent_control',tmp_path,tmp_path,tmp_path,20261010)
    assert len(rows)==4
    for lane in range(2):
        group=rows[lane::2];assert len({r['scenario'] for r in group})==1
        assert {r['method'] for r in group}=={'native','recent_no_score'}
        for row in group:
            assert '--wave2-steady-observer' not in row['cmd']
            if row['method']=='recent_no_score':assert row['cmd'][row['cmd'].index('--wave2-selector')+1]=='recent_no_score'
    full=build_wave2_cases(spec,'recent_hopper_control',tmp_path,tmp_path,tmp_path,20261010)
    assert len(full)==8
    for lane in range(2):assert {r['method'] for r in full[lane::2]}=={'native','steady_mass50','sum_fast','recent_no_score'}
