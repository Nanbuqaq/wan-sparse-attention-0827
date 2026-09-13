import json
from pathlib import Path
import pytest
from scripts.run_native_duration_wave import build_wave2_cases


def spec():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())


def test_complete_groups_use_same_pair_and_opposite_order(tmp_path):
    cases=build_wave2_cases(spec(),'matched_controls',tmp_path,tmp_path,tmp_path,20261010)
    assert len(cases)==10
    for lane in range(2):
        rows=cases[lane::2];assert len({r['scenario'] for r in rows})==1
        assert {r['method'] for r in rows}=={'native','steady_mass50','sum_old','sum_fast','sum_observer'}
        assert rows[-1]['method']=='sum_observer'
        for row in rows:
            assert '--wave2-capture' not in row['cmd']
            assert ('--wave2-steady-observer' in row['cmd'])==(row['method']=='sum_observer')
    assert [x['method'] for x in cases[0:8:2]]==list(reversed([x['method'] for x in cases[1:8:2]]))


def test_timing_three_alternating_pairs_and_own_native(tmp_path):
    rows=build_wave2_cases(spec(),'timing_repeats',tmp_path,tmp_path,tmp_path,20261010)
    assert len(rows)==18 and len({r['id'] for r in rows})==18
    for lane in range(2):
        cases=rows[lane::2]
        assert len({r['scenario'] for r in cases})==1
        for rep in range(3):
            order=[r['method'] for r in cases[rep*3:rep*3+3]]
            assert order==(['sum_old','sum_fast','native'] if rep%2==lane%2 else ['sum_fast','sum_old','native'])
        assert all(r['repeat_reason']=='timing_replication' and '--wave2-steady-observer' not in r['cmd'] for r in cases)


def test_local_serial_fallback_preserves_task_pairing_and_observer_dependencies(tmp_path):
    from scripts.run_native_duration_wave import serial_task_groups
    for stage in ('matched_controls','timing_repeats','scene_release'):
        original=build_wave2_cases(spec(),stage,tmp_path,tmp_path,tmp_path,20261010)
        serial=serial_task_groups(original);half=len(serial)//2
        assert len({r['scenario'] for r in serial[:half]})==1
        assert len({r['scenario'] for r in serial[half:]})==1
        assert all(r['cohort_pair']==0 for r in serial)
        for index,row in enumerate(serial):
            if 'reference_case_index' in row:
                ref=row['reference_case_index'];assert ref<index
                assert serial[ref]['method']=='sum_fast' and serial[ref]['scenario']==row['scenario']


def test_scene_diagnostic_retains_original_invalid_protocol_and_native_control(tmp_path):
    rows=build_wave2_cases(spec(),'scene_release',tmp_path,tmp_path,tmp_path,20261010)
    assert len(rows)==4
    assert {r['scenario'] for r in rows}=={'w2_ceramic_jug_revisit','w2_settled_pebble_bowl'}
    for lane in range(2):
        group=rows[lane::2];assert len({r['scenario'] for r in group})==1
        assert [r['method'] for r in group]==['native','scene_release']
        assert group[1]['cmd'][group[1]['cmd'].index('--wave2-method')+1]=='w2_scene_release'
        assert '--causal-scene-memory' not in group[1]['cmd']


def test_second_seed_recall_replication_is_fixed_and_paired(tmp_path):
    rows=build_wave2_cases(spec(),'recall_replication',tmp_path,tmp_path,tmp_path,20260914)
    assert len(rows)==8 and len({r['id'] for r in rows})==8
    for lane in range(2):
        task=rows[lane::2];assert len({r['scenario'] for r in task})==1
        assert {r['method'] for r in task}=={'native','steady_mass50','full_recall','steady_plus_recall'}
        for row in task:
            assert row['cmd'][row['cmd'].index('--seed')+1]=='20260914'
            assert '--wave2-preparation' not in row['cmd'] and '--wave2-steady-observer' not in row['cmd']
    assert [r['method'] for r in rows[0::2]]==list(reversed([r['method'] for r in rows[1::2]]))
    with pytest.raises(ValueError,match='seed20260914'):
        build_wave2_cases(spec(),'recall_replication',tmp_path,tmp_path,tmp_path,20260915)


@pytest.mark.parametrize('stage,scenario',[('recall_toy','generated_patchwork_toy_cut_revisit'),('recall_bead','generated_bead_state_cut_revisit')])
def test_regression_uses_existing_complete_cut_protocol(stage,scenario,tmp_path):
    rows=build_wave2_cases(spec(),stage,tmp_path,tmp_path,tmp_path,20260913)
    assert [r['method'] for r in rows]==['native','steady_mass50','full_recall','steady_plus_recall']
    for row in rows:
        assert row['scenario']==scenario
        assert row['cmd'][row['cmd'].index('--seed')+1]=='20260913'
        assert '--duration-probe-latents' not in row['cmd']
        assert '--wave2-preparation' not in row['cmd']
        assert '--wave2-route-audit' not in row['cmd']
