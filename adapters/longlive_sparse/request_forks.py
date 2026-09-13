"""Frozen development current-request interventions; not routing policy labels."""
REQUESTS={
    'update':'Back to the same clear cylindrical glass jar on the white table. The jar has now been emptied completely: there are no beads inside or falling into it. Preserve the same jar shape and show its empty transparent interior in a steady medium close-up.',
    'absent':'Back to the same white table. The glass jar has been removed, and no jar or beads remain. Show the empty tabletop in a steady medium close-up.'}

NEW_ROOM='The same ceramic jug now stands on a mossy stone ledge beside a forest path. Outdoor stone, moss and fern foliage fill the view. Keep its own handle, spout and original painted markings unchanged. Show the entire jug clearly in a realistic medium close-up.'


def same_subject_new_room(segments,prompts):
    start=segments[1]['start_latent']
    changed=[dict(segments[0]),dict(start_latent=start,scene_cut=True,role='same_subject_new_room',prompt=NEW_ROOM)]
    expanded=[list(prompts[0])]
    for frame in range(start,len(expanded[0])*8,8):
        expanded[0][frame//8]=('The scene transitions. ' if frame==start else '')+NEW_ROOM
    return changed,expanded


def current_return_fork(segments,prompts,request):
    if request not in ('keep','update','absent'):raise ValueError('unknown frozen current request')
    segments=[dict(s) for s in segments];prompts=[list(prompts[0])]
    if request=='keep':return segments,prompts
    start=segments[-1]['start_latent'];segments[-1]['prompt']=REQUESTS[request]
    for frame in range(start,len(prompts[0])*8,8):
        prompts[0][frame//8]=('The scene transitions. ' if frame==start else '')+REQUESTS[request]
    return segments,prompts
