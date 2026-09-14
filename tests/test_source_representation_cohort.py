import json
from pathlib import Path
from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices


def test_context_pairs_are_complete_and_extra_compute_is_not_hidden(tmp_path):
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'source_representation',tmp_path,tmp_path,tmp_path,20261023)
    assert len(rows)==12 and list(map(len,case_lane_indices(rows,4)))==[3]*4
    for indices in case_lane_indices(rows,4):
        assert {rows[i]['method'] for i in indices}=={'raw_record','past_reencode','current_reencode'}
        assert len({rows[i]['cmd'][rows[i]['cmd'].index('--seed')+1] for i in indices})==1
    for r in rows:
        cmd=r['cmd'];assert '--audit-shared-conditioning-inputs' in cmd
        assert '--source-layer-stream' not in cmd and '--source-clean-cache-witness' not in cmd
        assert cmd[cmd.index('--source-snapshot-window')+1]=='latest8'
        assert cmd[cmd.index('--source-stage-policy')+1]=='all'
        assert r['auxiliary_compute_is_additional'] and r['raw_archives_still_retained']
