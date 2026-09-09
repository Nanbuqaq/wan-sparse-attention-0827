import math

import pytest
import torch

from adapters.longlive_sparse.native_scene_admission import SceneDescriptor,choose_scene,has_revisit_cue


def descriptor(version,end,similarity):
    return SceneDescriptor(version,end,8.,torch.tensor([similarity,math.sqrt(1-similarity**2)]))


def test_latest_state_over_nearby_initial_similarity_without_reading_KV():
    candidates=[descriptor(1,24,.831),descriptor(2,48,.816),descriptor(3,96,.99)]
    result=choose_scene('The scene transitions. Back to the same jar.',torch.tensor([1.,0.]),candidates,96)
    assert result['selected_version']==2
    assert len(result['scores'])==2


def test_explicitly_unrelated_conditions_and_no_revisit_cue_do_not_admit():
    candidates=[descriptor(1,24,.7)]
    assert choose_scene('Back to the spaceship.',torch.tensor([1.,0.]),candidates,96)['selected_version'] is None
    assert choose_scene('A new jar appears.',torch.tensor([1.,0.]),[descriptor(1,24,.99)],96)['selected_version'] is None


def test_current_condition_changes_the_choice_and_future_candidates_are_rejected():
    candidates=[descriptor(1,24,.99),descriptor(2,48,.5)]
    assert choose_scene('Return to the empty jar.',torch.tensor([1.,0.]),candidates,96)['selected_version']==1
    with pytest.raises(ValueError):choose_scene('Back to it.',torch.tensor([1.,0.]),[descriptor(3,104,.99)],96)


def test_cue_is_current_instruction_not_an_unanchored_keyword():
    assert has_revisit_cue('The camera returns to the toy.')
    assert not has_revisit_cue('Do not return to the toy.')


def test_latest_candidate_must_still_pass_the_absolute_floor():
    result=choose_scene('Back to the jar.',torch.tensor([1.,0.]),[descriptor(1,24,.81),descriptor(2,48,.79)],96)
    assert result['selected_version']==1
