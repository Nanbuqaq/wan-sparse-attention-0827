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
