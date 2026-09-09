import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_longlive2_native_reference import native_cut_schedule,episode_source_and_target

ROOT=Path(__file__).resolve().parents[1]


def test_new_state_has_longer_hold_and_a_qualified_long_absence():
    segments,prompts=native_cut_schedule(ROOT,'blue_canvas_revisit')
    assert [s['start_latent'] for s in segments]==[0,16,32,64,96]
    assert episode_source_and_target(segments)==(64,96)
    assert [i for i,p in enumerate(prompts[0]) if p.startswith('The scene transitions. ')]==[2,8,12]
    assert len(set(prompts[0][4:8]))==1
    assert 'blue' not in prompts[0][12].lower()
    _,visible=native_cut_schedule(ROOT,'blue_canvas_visible_control')
    assert prompts[0][:8]==visible[0][:8]
    with pytest.raises(ValueError):native_cut_schedule(ROOT,'blue_canvas_revisit',gate=True)


def test_causal_policy_is_frozen_before_new_state_screen():
    config=json.loads((ROOT/'configs/system/native_blue_canvas_screen.json').read_text())
    assert config['seeds']==[20260923,20260924] and not config['memory_interventions_allowed']
    lock=config['selection_policy_frozen_before_screen']
    for filename,key in [('native_scene_admission.py','scene_admission_sha256'),('native_causal_scene_memory.py','causal_memory_sha256')]:
        assert hashlib.sha256((ROOT/'adapters/longlive_sparse'/filename).read_bytes()).hexdigest()==lock[key]


@pytest.mark.parametrize('base,variant',[('blue_canvas_revisit','blue_canvas_positive_stop_revisit'),
    ('blue_canvas_visible_control','blue_canvas_positive_stop_visible_control')])
def test_positive_stop_changes_only_the_actor_clause_without_cut_or_color_restatement(base,variant):
    a,pa=native_cut_schedule(ROOT,base);b,pb=native_cut_schedule(ROOT,variant)
    assert [s['start_latent'] for s in a]==[s['start_latent'] for s in b]
    assert pa[0][:4]==pb[0][:4] and pa[0][8:]==pb[0][8:]
    for i in range(4,8):
        assert pa[0][i].removeprefix('The painter and roller have left the view. ')==pb[0][i].removeprefix('Only the canvas and its wooden easel are visible in the quiet studio. ')
        assert 'blue' not in pb[0][i].lower()
    assert [i for i,p in enumerate(pb[0]) if p.startswith('The scene transitions. ')]==[2,8,12]
