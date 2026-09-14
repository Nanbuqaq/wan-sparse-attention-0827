"""Native role/provenance controls at a declared return transition."""
POLICIES=('full','pin_first','recent_first','both_first','anchor_transition')


def permitted_native_frames(policy,owners,physical,roles,phase,*,first_return,returning,global_slots):
    if policy not in POLICIES:raise ValueError('unknown return context policy')
    active=returning and (first_return or policy=='anchor_transition')
    if policy=='full' or not active:return list(range(len(physical)))
    keep=[]
    for position,(slot,role) in enumerate(zip(physical,roles)):
        owner=owners[slot]
        if owner[0]!='native':raise ValueError('side-reader native window unexpectedly contains installed source')
        mandatory=slot<global_slots or role['current']
        stale=owner[3]!=phase
        drop=stale and not mandatory and (policy in ('both_first','anchor_transition')
             or (policy=='pin_first' and role['pin']) or (policy=='recent_first' and not role['pin']))
        if not drop:keep.append(position)
    return keep


def contiguous_frame_runs(positions):
    if not positions:raise ValueError('current/native context cannot be empty')
    runs=[];start=previous=positions[0]
    for value in positions[1:]:
        if value!=previous+1:runs.append((start,previous+1));start=value
        previous=value
    runs.append((start,previous+1))
    return runs
