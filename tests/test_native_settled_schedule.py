from pathlib import Path

import pytest

from scripts.run_longlive2_native_reference import native_cut_schedule

ROOT=Path(__file__).resolve().parents[1]


def test_settled_screen_has_two_hold_chunks_and_only_declared_cuts():
    segments,prompts=native_cut_schedule(ROOT,'settled_bead_revisit')
    assert [s['start_latent'] for s in segments]==[0,16,32,48,96]
    assert len(prompts[0])==16
    assert [i for i,p in enumerate(prompts[0]) if p.startswith('The scene transitions. ')]==[2,6,12]
    assert prompts[0][4]==prompts[0][5] and 'completely stopped' in prompts[0][5]
    _,control=native_cut_schedule(ROOT,'settled_bead_visible_control')
    assert prompts[0][:6]==control[0][:6] and prompts[0][6]!=control[0][6]


def test_old_protocol_remains_identical():
    segments,prompts=native_cut_schedule(ROOT,'generated_bead_state_cut_revisit')
    assert [s['start_latent'] for s in segments]==[0,24,48,96]
    assert [i for i,p in enumerate(prompts[0]) if p.startswith('The scene transitions. ')]==[3,6,12]


def test_new_feasibility_is_not_shrunk_to_invalid_low_resolution_gate():
    with pytest.raises(ValueError):native_cut_schedule(ROOT,'settled_bead_revisit',gate=True)
