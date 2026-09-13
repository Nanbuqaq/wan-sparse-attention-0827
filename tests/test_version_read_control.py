from adapters.longlive_sparse.version_read_control import permitted_version_positions


def test_fixed_read_never_removes_native_current_and_expires_with_physical_owner():
    old=('recalled',20,2,'binding',0.,88,24.);new=('recalled',44,2,'binding',8.,92,24.)
    current=('native',96,20,24.);global_owner=('native',0,1,0.)
    owners=[global_owner,old,new,current];mapping={old:1,new:2}
    assert permitted_version_positions(owners,list(range(4)),mapping,'old')==[0,1,3]
    assert permitted_version_positions(owners,list(range(4)),mapping,'new')==[0,2,3]
    owners[1]=('native',104,21,24.)
    assert permitted_version_positions(owners,list(range(4)),mapping,'new')==[0,1,2,3]
