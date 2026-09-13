from pathlib import Path
from scripts.run_longlive2_native_reference import native_cut_schedule
from adapters.longlive_sparse.request_forks import current_return_fork,REQUESTS,same_subject_new_room


def test_current_request_changes_only_return_conditions():
    segments,prompts=native_cut_schedule(Path(__file__).resolve().parents[1],'generated_bead_state_cut_revisit')
    for request in ('keep','update','absent'):
        changed,expanded=current_return_fork(segments,prompts,request)
        assert expanded[0][:12]==prompts[0][:12] and changed[:-1]==segments[:-1]
        assert [s['start_latent'] for s in changed]==[0,24,48,96]
        if request=='keep':assert expanded==prompts
        else:assert expanded[0][12]=='The scene transitions. '+REQUESTS[request]
    assert 'emptied completely' not in prompts[0][-1]


def test_new_room_keeps_source_prefix_and_has_no_future_return():
    segments,prompts=native_cut_schedule(Path(__file__).resolve().parents[1],'w2_ceramic_jug_revisit')
    changed,expanded=same_subject_new_room(segments,prompts)
    assert expanded[0][:6]==prompts[0][:6] and len(changed)==2
    assert [i for i,p in enumerate(expanded[0]) if p.startswith('The scene transitions. ')]==[6]
    assert all('Back to' not in p for p in expanded[0][6:])
