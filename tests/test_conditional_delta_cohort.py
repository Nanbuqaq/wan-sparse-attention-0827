import json
from pathlib import Path
from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices


def test_each_pair_has_native_raw_and_two_matched_direction_controls(tmp_path):
    root=Path(__file__).resolve().parents[1];spec=json.loads((root/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'conditional_delta',tmp_path,tmp_path,tmp_path,20261023)
    assert len(rows)==17 and list(map(len,case_lane_indices(rows,4)))==[5,4,4,4]
    for lane,ids in enumerate(case_lane_indices(rows,4)):
        assert {rows[i]['method'] for i in ids}==({'native','raw_record','null','forward','reverse'} if lane==0 else {'native','raw_record','forward','reverse'})
        assert len({rows[i]['cmd'][rows[i]['cmd'].index('--seed')+1] for i in ids})==1
        for i in ids:
            r=rows[i];cmd=r['cmd'];assert ('--state-quarter-turn' in cmd)==(lane==1)
            assert '--audit-shared-conditioning-inputs' in cmd
            assert ('--source-conditional-delta' in cmd)==(r['method'] in ('forward','reverse','null'))
            if r['method']=='native':assert '--source-representation' not in cmd
            else:assert cmd[cmd.index('--source-representation')+1]=='raw_record'

    null=rows[2];assert null['method']=='null' and null['reference_case_index']==1
    assert null['cmd'][null['cmd'].index('--equivalence-reference')+1]==str(tmp_path/rows[1]['id']/'summary.json')
