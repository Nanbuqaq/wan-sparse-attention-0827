import json
from pathlib import Path
import pytest
from adapters.longlive_sparse.state_update_protocol import state_update_schedule
from adapters.longlive_sparse.past_appearance_control import append_past_appearance
from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices
ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('task',['red_toolbox','silver_case'])
@pytest.mark.parametrize('operation',['keep','close'])
def test_text_control_only_restates_a_bounded_past_span_at_return(task,operation):
    name=f'w2_state_{task}_{operation}';s,p=state_update_schedule(ROOT,name)
    changed,text,a=append_past_appearance(ROOT,name,s,p)
    assert text[0][:12]==p[0][:12] and changed[:-1]==s[:-1]
    assert a['clause'] in s[1]['prompt'] and a['clause_UTF8_bytes']<=512
    assert text[0][12].startswith(p[0][12])
    assert 'fully raised' not in a['clause'] and 'fully shut' not in a['clause']
    assert a['manually_selected_identity_clause'] and not a['autonomous_extraction_claim']


def test_representation_cohort_never_gives_text_controls_a_free_raw_archive(tmp_path):
    spec=json.loads((ROOT/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'state_representation',tmp_path,tmp_path,tmp_path,20261023)
    assert len(rows)==12 and list(map(len,case_lane_indices(rows,2)))==[6,6]
    for r in rows:
        cmd=r['cmd']
        if r['method'] in ('filter_no_archive','past_text_filter'):
            assert '--source-no-archive' in cmd
            assert cmd[cmd.index('--source-lifetime-policy')+1]=='off'
        if r['method'].startswith('raw_'):
            assert '--source-no-archive' not in cmd and '--state-past-appearance-text' not in cmd
