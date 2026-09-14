"""Exact-budget diversity with native/strong fast and a late native timing control."""


def build_information_groups(spec,assets,source,output,seed,base_builder):
    if seed!=20261010:raise ValueError('first information-group cohort fixes existing development seed')
    base=base_builder(spec,'native',assets,source,output,seed);rows=[]
    for lane,task in enumerate(('w2_rotating_wooden_bird','w2_tracking_delivery_cart')):
        template=next(x for x in base if x['scenario']==task)
        for kind in ('native','old_block64_fast','flat_exact','physical16','value16','native_late'):
            name=f'{task}__s{seed}__information_{kind}';cmd=list(template['cmd'])
            def put(key,value):
                if key in cmd:cmd[cmd.index(key)+1]=str(value)
                else:cmd.extend([key,str(value)])
            put('--output',output/name)
            if not kind.startswith('native'):
                put('--wave2-method','w2_steady_sparse');put('--wave2-selector','query_sum_batch4')
                put('--wave2-preparation','geometry_cache');cmd+=['--wave2-age-observer']
            if kind in ('flat_exact','physical16','value16'):put('--wave2-information-groups',kind)
            rows.append(dict(id=name,scenario=task,method=kind,cmd=cmd,latent_frames=128,cohort_pair=lane,
                repeat_reason='warm_native_timing_control_after_documented_startup_confound' if kind=='native_late'
                    else 'new_bounded_information_diversity_and_exact_resource_controls',formal_holdout=False))
    return rows
