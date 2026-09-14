"""Frozen return-motion protocol; no future event labels reach the reader."""
TOY_RETURN_MOTION=('Back to the very same handmade mechanical toy on the white table beside its gray gift box. '
    'The toy slowly turns through a quarter turn beside the box, continuing the rotation smoothly across the shot. '
    'Preserve its original face, colors, markings and construction while showing the changing view of the whole toy.')


def lifetime_schedule(segments,prompts,*,motion=False,gate=False):
    segments=[dict(s) for s in segments];prompts=[list(prompts[0])]
    if motion:
        segments[-1]['prompt']=TOY_RETURN_MOTION
        first=segments[-1]['start_latent']//8
        for i in range(first,len(prompts[0])):
            prompts[0][i]=('The scene transitions. ' if i==first else '')+TOY_RETURN_MOTION
    if gate:
        if len(prompts[0])!=8 or segments[-1]['start_latent']!=48:
            raise ValueError('source gate extends the qualified64 layout return only')
        prompts[0].extend([prompts[0][-1]]*2)
    if len(prompts[0])*8-segments[-1]['start_latent']<32:
        raise ValueError('need three source chunks plus one source-off chunk')
    return segments,prompts
