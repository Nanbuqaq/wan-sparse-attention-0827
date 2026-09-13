from collections import Counter
from adapters.longlive_sparse.source_layer_budget import source_frame_slots,LAYERRECALL_PRIOR
import json
from pathlib import Path
from scripts.run_native_duration_wave import build_wave2_cases,case_lane_indices


def test_layer_controls_match_total_and_each_original_frame_exposure():
    for policy in ('uniform_third','early10','late10','prior10'):
        counts=Counter(i for layer in range(30) for i in source_frame_slots(layer,policy))
        assert counts=={i:10 for i in range(8)}
        assert all(source_frame_slots(layer,policy,True)==tuple(range(8)) for layer in range(30))
    assert [i for i in range(30) if source_frame_slots(i,'prior10')]==list(LAYERRECALL_PRIOR)


def test_layer_cohort_does_not_mix_weighting_or_other_new_interventions(tmp_path):
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'source_layers',tmp_path,tmp_path,tmp_path,20261010)
    assert len(rows)==12 and list(map(len,case_lane_indices(rows,2)))==[6,6]
    assert all('--source-memory-beta' not in r['cmd'] and '--wave2-version-policy' not in r['cmd'] for r in rows)
