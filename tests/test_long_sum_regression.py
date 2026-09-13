import json
from pathlib import Path
import pytest
from adapters.longlive_sparse.native_duration_probe import extend_constant_schedule
from scripts.run_native_duration_wave import build_wave2_cases


def test_long_continuous_schedule_has_no_new_cuts_or_duplicated_history():
    segments=[dict(start_latent=0,prompt='rotate',scene_cut=False)]
    out,prompts=extend_constant_schedule(segments,[['rotate']*16],728,128)
    assert len(prompts[0])==91 and out==segments
    assert prompts[0][:16]==['rotate']*16
    with pytest.raises(ValueError):extend_constant_schedule(segments,[['The scene transitions. rotate']*16],728,128)


def test_three_arm_long_regression_keeps_native_and_whole_output_reference(tmp_path):
    spec=json.loads((Path(__file__).resolve().parents[1]/'configs/system/wave2_scenarios.json').read_text())
    rows=build_wave2_cases(spec,'long_sum_regression',tmp_path,tmp_path,tmp_path,20261010)
    assert [r['method'] for r in rows]==['native','sum_old','sum_fast']
    assert all(r['latent_frames']==728 for r in rows)
    assert all(r['cmd'][r['cmd'].index('--duration-probe-latents')+1]=='728' for r in rows)
    assert rows[-1]['reference_case_index']==1
    assert all('--wave2-steady-observer' not in r['cmd'] for r in rows)
