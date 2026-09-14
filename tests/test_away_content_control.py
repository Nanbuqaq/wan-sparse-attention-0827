from pathlib import Path
import pytest
from adapters.longlive_sparse.away_content_control import replace_away_content, PENDULUM
from adapters.longlive_sparse.state_update_protocol import state_update_schedule

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('scenario', ['w2_state_pattern_tile_keep','w2_state_red_toolbox_last_open'])
@pytest.mark.parametrize('gate', [False,True])
def test_only_away_changes_and_causal_boundaries_stay_fixed(scenario,gate):
    segments,prompts=state_update_schedule(ROOT,scenario,gate=gate,episode_gate=gate)
    changed,output,audit=replace_away_content(segments,prompts,'pendulum')
    ids=audit['changed_blocks']
    assert [i for i,(a,b) in enumerate(zip(prompts[0],output[0])) if a!=b]==ids
    assert [i for i,p in enumerate(prompts[0]) if p.startswith('The scene transitions. ')]==[
        i for i,p in enumerate(output[0]) if p.startswith('The scene transitions. ')]
    assert all(a==b for a,b in zip(segments,changed) if a['role']!='away')
    assert segments!=changed and prompts[0][ids[0]]!=output[0][ids[0]]
    assert all(p.removeprefix('The scene transitions. ')==PENDULUM for p in output[0][ids[0]:ids[-1]+1])


def test_no_silent_intervention_on_unknown_schedule():
    with pytest.raises(ValueError):replace_away_content([], [['example']], 'pendulum')
