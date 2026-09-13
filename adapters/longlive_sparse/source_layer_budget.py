"""Frozen source-layer controls, with equal aggregate exposure per source frame."""
LAYERRECALL_PRIOR=(4,9,10,12,13,15,16,17,18,26)


def source_frame_slots(layer,policy,clean=False):
    if not 0<=layer<30:raise ValueError('native30-layer control required')
    if policy not in ('full','uniform_third','early10','late10','prior10'):raise ValueError('unknown layer policy')
    if clean or policy=='full':return tuple(range(8))
    if policy=='early10':return tuple(range(8)) if layer<10 else ()
    if policy=='late10':return tuple(range(8)) if layer>=20 else ()
    if policy=='prior10':return tuple(range(8)) if layer in LAYERRECALL_PRIOR else ()
    # 10 layers use2 frames and20 use3: total80, each of8 frames appears10 times.
    quotas=[2 if i%3==0 else 3 for i in range(30)];offset=sum(quotas[:layer])
    return tuple((offset+i)%8 for i in range(quotas[layer]))
