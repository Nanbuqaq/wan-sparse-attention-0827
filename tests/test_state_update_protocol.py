import json
from pathlib import Path

from adapters.longlive_sparse.state_update_protocol import state_update_schedule
from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices

ROOT=Path(__file__).resolve().parents[1]


def test_native_keep_and_close_share_all_prior_prompts_and_only_one_return_cut():
    for task in ('blue_box','silver_case'):
        a,pa=state_update_schedule(ROOT,f'w2_state_{task}_keep')
        b,pb=state_update_schedule(ROOT,f'w2_state_{task}_close')
        assert a[:-1]==b[:-1] and pa[0][:12]==pb[0][:12]
        assert len(pa[0])==len(pb[0])==16 and pa[0][12]!=pb[0][12]
        assert [i for i,t in enumerate(pa[0]) if t.startswith('The scene transitions. ')]==[1,6,12]
        assert pa[0][12:]==[pa[0][12]]+[pa[0][13]]*3


def test_state_cohort_is_native_only_and_task_pairs_remain_complete(tmp_path):
    spec=json.loads((ROOT/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'state_feasibility',tmp_path,tmp_path,tmp_path,20261021)
    assert len(rows)==4 and list(map(len,case_lane_indices(rows,2)))==[2,2]
    assert all(r['cmd'][r['cmd'].index('--wave2-method')+1]=='w2_native' for r in rows)
    assert all('--source-lifetime-policy' not in r['cmd'] for r in rows)
