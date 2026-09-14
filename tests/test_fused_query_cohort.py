import json
from pathlib import Path
from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices


def test_exact_graph_dependencies_stay_on_same_physical_pair(tmp_path):
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'fused_query_system',tmp_path,tmp_path,tmp_path,20261010)
    assert len(rows)==16 and list(map(len,case_lane_indices(rows,2)))==[8,8]
    for i,row in enumerate(rows):
        if 'reference_case_index' in row:
            j=row['reference_case_index'];assert j<i
            assert rows[j]['scenario']==row['scenario'] and rows[j]['cohort_pair']==row['cohort_pair']
            ref=Path(row['cmd'][row['cmd'].index('--equivalence-reference')+1])
            assert ref==tmp_path/rows[j]['id']/'summary.json'
    for indices in case_lane_indices(rows,2):
        methods=[rows[i]['method'] for i in indices]
        assert methods[3:7]==['split_specific_torch','split_specific_fused','split_specific_fused','split_specific_torch']
        assert methods[0]==methods[-1]=='native_torch'
