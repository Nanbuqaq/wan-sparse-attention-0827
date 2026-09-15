import json
from pathlib import Path
from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices


def test_two_new_request_contexts_have_complete_strong_controls(tmp_path):
    root=Path(__file__).resolve().parents[1];spec=json.loads((root/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'global_payload_replication',tmp_path,tmp_path,tmp_path,20261023)
    assert list(map(len,case_lane_indices(rows,2)))==[5,5]
    for lane,indices in enumerate(case_lane_indices(rows,2)):
        assert {rows[i]['method'] for i in indices}=={'native','original','zero_v','zero_kv','mean_v'}
        for i in indices:
            cmd=rows[i]['cmd'];assert ('--state-quarter-turn' in cmd)==(lane==1)
            if rows[i]['method']=='native':assert '--global-payload' not in cmd
            else:assert '--archive-skip-resident-global' in cmd and cmd[cmd.index('--archive-write-backend')+1]=='sync'
