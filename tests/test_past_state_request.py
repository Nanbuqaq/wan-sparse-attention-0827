from pathlib import Path
from adapters.longlive_sparse.state_update_protocol import state_update_schedule
from adapters.longlive_sparse.past_appearance_control import append_past_state_request


def test_old_text_control_preserves_prefix_and_does_not_invent_observed_state():
    root=Path(__file__).resolve().parents[1]
    s,p=state_update_schedule(root,'w2_state_red_toolbox_last_open')
    changed,text,audit=append_past_state_request(root,'w2_state_red_toolbox_last_open',s,p)
    assert changed[:-1]==s[:-1] and text[0][:12]==p[0][:12]
    assert s[2]['prompt'] in audit['clause'] and audit['clause_UTF8_bytes']<=1024
    assert audit['past_requested_not_observed_state'] and not audit['autonomous_extraction_claim']
