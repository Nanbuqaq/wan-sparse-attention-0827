from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_longlive2_native_reference import native_cut_schedule,episode_source_and_target,reviewed_settled_memory_protocol

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


@pytest.mark.parametrize('scenario',['generated_bead_state_cut_revisit','generated_patchwork_toy_cut_revisit','settled_bead_revisit'])
def test_named_away_boundary_preserves_old_source_and_selects_actual_settled_source(scenario):
    segments,_=native_cut_schedule(ROOT,scenario)
    assert episode_source_and_target(segments)==(48,96)
    if scenario=='settled_bead_revisit':assert segments[2]['start_latent']==32


def test_old_short_gate_named_boundaries_are_unchanged():
    segments,_=native_cut_schedule(ROOT,'generated_bead_state_cut_revisit',gate=True,episode_gate=True)
    assert episode_source_and_target(segments)==(16,48)


def test_visible_control_is_not_a_hidden_long_absence_episode():
    segments,_=native_cut_schedule(ROOT,'settled_bead_visible_control')
    with pytest.raises(ValueError):episode_source_and_target(segments)


def protocol_args(**overrides):
    values=dict(reviewed_memory_protocol='settled_state_v1',cut_scenario='settled_bead_revisit',seed=20260919,
        gate=False,episode_memory_mode='raw_reveal',episode_destination='shot',episode_position_policy='recent_virtual',
        scene_context_reset=False,memory_reconstruction='none',episode_restore_after_frames=0,
        cut_component_ablation='none',native_local_frames=32,cfg1_positive_cache_only=True)
    return SimpleNamespace(**dict(values,**overrides))


def test_reviewed_source_protocol_is_explicit_and_hash_locked():
    report=reviewed_settled_memory_protocol(ROOT,protocol_args())
    assert len(report['config_sha256'])==64 and report['spec']['source_frames']==list(range(40,48))
    assert reviewed_settled_memory_protocol(ROOT,SimpleNamespace(reviewed_memory_protocol=None)) is None


@pytest.mark.parametrize('change',[
    dict(seed=20260921),dict(cut_scenario='settled_bead_visible_control'),dict(gate=True),
    dict(episode_memory_mode='raw_away'),dict(episode_destination='global'),dict(scene_context_reset=True),
    dict(episode_position_policy='phase_only'),dict(native_local_frames=128),dict(cfg1_positive_cache_only=False)])
def test_reviewed_protocol_does_not_unlock_unreviewed_experiments(change):
    with pytest.raises(ValueError):reviewed_settled_memory_protocol(ROOT,protocol_args(**change))
