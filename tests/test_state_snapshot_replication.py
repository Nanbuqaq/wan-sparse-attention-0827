import json
from pathlib import Path
from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices
from adapters.longlive_sparse.state_update_protocol import state_update_schedule

ROOT=Path(__file__).resolve().parents[1]


def test_replication_keeps_paired_inputs_and_does_not_promote_unreviewed_source(tmp_path):
    spec=json.loads((ROOT/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'state_snapshot_replication',tmp_path,tmp_path,tmp_path,20261022)
    assert len({r['id'] for r in rows})==10
    assert list(map(len,case_lane_indices(rows,2)))==[5,5]
    new=[r for r in rows if 'red_toolbox' in r['scenario']]
    assert len(new)==2 and all(r['cmd'][r['cmd'].index('--wave2-method')+1]=='w2_native' for r in new)
    for operation in ('keep','close'):
        task=[r for r in rows if r['scenario']==f'w2_state_silver_case_{operation}']
        assert len(task)==4
        assert {r['cmd'][r['cmd'].index('--seed')+1] for r in task}=={'20261022'}
    a,pa=state_update_schedule(ROOT,'w2_state_red_toolbox_keep')
    b,pb=state_update_schedule(ROOT,'w2_state_red_toolbox_close')
    assert pa[0][:12]==pb[0][:12] and pa[0][12:]!=pb[0][12:]
    assert a[:3]==b[:3]
