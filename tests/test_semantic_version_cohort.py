import json
from pathlib import Path
from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices


def test_semantic_wave_freezes_one_seed_and_own_native_controls(tmp_path):
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'semantic_versions',tmp_path,tmp_path,tmp_path,20260913)
    assert len(rows)==12 and [len(x) for x in case_lane_indices(rows,2)]==[6,6]
    for r in rows:
        assert r['cmd'][r['cmd'].index('--seed')+1]=='20260913'
    bead=[r for r in rows if 'bead' in r['scenario']]
    assert {r['method'] for r in bead}=={'keep_native','keep_latest8','keep_old4_new4','keep_uniform8','update_native','update_latest8'}
