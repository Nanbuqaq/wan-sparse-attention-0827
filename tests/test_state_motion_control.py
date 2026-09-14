from pathlib import Path
from adapters.longlive_sparse.state_update_protocol import state_update_schedule
from adapters.longlive_sparse.state_motion_control import append_quarter_turn


def test_motion_only_changes_arrived_return_request():
    root=Path(__file__).resolve().parents[1]
    segments,prompts=state_update_schedule(root,'w2_state_red_toolbox_keep')
    changed,out,audit=append_quarter_turn('w2_state_red_toolbox_keep',segments,prompts)
    assert changed[:-1]==segments[:-1] and out[0][:12]==prompts[0][:12]
    assert audit['changed_blocks']==[12,13,14,15]
    assert all(a in b and a!=b for a,b in zip(prompts[0][12:],out[0][12:]))


def test_motion_appearance_control_uses_only_exact_past_text_not_observed_detail():
    from adapters.longlive_sparse.past_appearance_control import append_past_appearance
    root=Path(__file__).resolve().parents[1]
    segments,prompts=state_update_schedule(root,'w2_state_red_toolbox_keep')
    moving,texts,_=append_quarter_turn('w2_state_red_toolbox_keep',segments,prompts)
    result,combined,audit=append_past_appearance(root,'w2_state_red_toolbox_keep',moving,texts)
    assert combined[0][:12]==prompts[0][:12] and result[:-1]==segments[:-1]
    assert audit['clause'] in segments[1]['prompt']
    assert audit['observed_identity_or_state_certificate'] is False
    assert all(old in new for old,new in zip(texts[0][12:],combined[0][12:]))
