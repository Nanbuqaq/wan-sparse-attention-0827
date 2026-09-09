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


def test_nocut_anaphora_changes_only_declared_cut_prefixes():
    _,base=native_cut_schedule(ROOT,'settled_bead_visible_control')
    _,control=native_cut_schedule(ROOT,'settled_bead_nocut_anaphora')
    assert base[0][:6]==control[0][:6]
    for i in range(6,16):assert base[0][i].removeprefix('The scene transitions. ')==control[0][i]
    assert [i for i,p in enumerate(control[0]) if p.startswith('The scene transitions. ')]==[2]


def test_explicit_continuation_repeats_the_already_active_hold_condition():
    _,prompts=native_cut_schedule(ROOT,'settled_bead_nocut_explicit')
    assert len(set(prompts[0][4:]))==1 and 'red glass beads' in prompts[0][4]
