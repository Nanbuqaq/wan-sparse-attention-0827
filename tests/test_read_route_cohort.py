import json
from pathlib import Path
from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices


def test_read_route_has_constant_read_policies_and_two_seed_controls(tmp_path):
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'read_and_route',tmp_path,tmp_path,tmp_path,20261010)
    assert len(rows)==16 and [len(x) for x in case_lane_indices(rows,2)]==[8,8]
    assert {r['method'] for r in rows if 'bead' in r['scenario']}=={f'{request}_{method}' for request in ('keep','update') for method in ('native','all','old','new')}
    assert all('--wave2-age-observer' not in r['cmd'] for r in rows if 'bead' in r['scenario'])
