from pathlib import Path
import pytest
from scripts.run_native_duration_wave import build_duration_cases


def test_matched_noise_four_case_batch_has_unique_identity_and_two_method_controls():
    cases=build_duration_cases(scenarios=['generated_patchwork_toy_cut_revisit'],lengths=[128,728],seed=20261002,
        alignment='return_event',assets=Path('/assets'),source=Path('/source'),output=Path('/out'))
    assert len(cases)==4 and len({c['id'] for c in cases})==4
    assert {(c['latent_frames'],c['method']) for c in cases}=={(n,m) for n in (128,728) for m in ('native','scene_full')}
    for c in cases:
        assert '--duration-noise-alignment' in c['cmd']
        assert ('--causal-scene-memory' in c['cmd'])==(c['method']=='scene_full')
    with pytest.raises(ValueError):
        build_duration_cases(scenarios=['x'],lengths=[128,128],seed=1,alignment='absolute',assets=Path('/a'),source=Path('/s'),output=Path('/o'))
