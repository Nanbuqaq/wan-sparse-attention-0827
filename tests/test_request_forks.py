from pathlib import Path
from scripts.run_longlive2_native_reference import native_cut_schedule
from adapters.longlive_sparse.request_forks import current_return_fork,REQUESTS


def test_current_request_changes_only_return_conditions():
    segments,prompts=native_cut_schedule(Path(__file__).resolve().parents[1],'generated_bead_state_cut_revisit')
    for request in ('keep','update','absent'):
        changed,expanded=current_return_fork(segments,prompts,request)
        assert expanded[0][:12]==prompts[0][:12] and changed[:-1]==segments[:-1]
        assert [s['start_latent'] for s in changed]==[0,24,48,96]
        if request=='keep':assert expanded==prompts
        else:assert expanded[0][12]=='The scene transitions. '+REQUESTS[request]
    assert 'emptied completely' not in prompts[0][-1]
