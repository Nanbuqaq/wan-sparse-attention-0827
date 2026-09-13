import json
from pathlib import Path
from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices
from scripts.run_longlive2_native_reference import native_cut_schedule


def test_multievent_schedule_and_budget_pairs(tmp_path):
    root=Path(__file__).resolve().parents[1]
    segments,prompts=native_cut_schedule(root,'w2_multi_event')
    assert [s['start_latent'] for s in segments]==[0,8,32,64,88,120] and len(prompts[0])==16
    assert prompts[0][8].startswith('The scene transitions. Back to')
    assert prompts[0][15].startswith('The scene transitions. Back to')
    spec=json.loads((root/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'multi_event',tmp_path,tmp_path,tmp_path,20260913)
    assert len(rows)==8 and list(map(len,case_lane_indices(rows,2)))==[4,4]
    assert all('--scene-payload-catalog' in r['cmd'] for r in rows if r['method'].startswith('restore'))
    lineage=build_wave2_cases(spec,'lineage_controls',tmp_path,tmp_path,tmp_path,20260913)
    assert len(lineage)==10 and list(map(len,case_lane_indices(lineage,2)))==[5,5]
    assert {r['method'] for r in lineage}=={'raw_latest','canonical_latest','raw_max','canonical_max','canonical_max6'}
