import pytest
from adapters.longlive_sparse.access_motion_memory import decode_owner,eligible_positions


def test_typed_phase_and_admitted_version_are_distinct():
    native=('native',0,1,0.)
    source=('recalled',40,2,'new-binding',8.,88,24.)
    other=('recalled',40,1,'old-binding',8.,88,24.)
    current=('native',96,10,24.)
    owners=[native,source,other,current]
    assert decode_owner(source)['phase']==24. and decode_owner(source)['source_phase']==8.
    assert eligible_positions(owners,list(range(4)),24.,True,{source})==[1,3]
    assert eligible_positions(owners,list(range(4)),24.,True,set())==[3]
    assert eligible_positions(owners,list(range(4)),24.,False,set())==[0,1,2,3]
    # Same source scene's global anchor is not implicitly admitted.
    owners[0]=('native',32,1,8.)
    assert eligible_positions(owners,list(range(4)),24.,True,{source})==[1,3]


def test_admission_without_physical_residency_does_not_add_a_token():
    source=('recalled',40,2,'binding',8.,88,24.)
    current=('native',104,12,24.)
    assert eligible_positions([current],[0],24.,True,{source})==[0]
    for bad in (None,('recalled',40,2,8.),('native',1)):
        with pytest.raises(ValueError):decode_owner(bad)
