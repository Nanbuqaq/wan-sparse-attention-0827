"""Native same-prefix keep/close feasibility before state-admission algorithms."""


def build_state_feasibility(spec,assets,source,output,seed,base_builder):
    if seed!=20261021:raise ValueError('state feasibility seed is frozen before new videos')
    template=base_builder(spec,'native',assets,source,output,spec['development_seed'])[0];rows=[]
    for lane,task in enumerate(('blue_box','silver_case')):
        for operation in ('keep','close'):
            scenario=f'w2_state_{task}_{operation}';name=f'{scenario}__s{seed}__native'
            cmd=list(template['cmd'])
            def put(key,value):
                if key in cmd:cmd[cmd.index(key)+1]=str(value)
                else:cmd.extend([key,str(value)])
            put('--output',output/name);put('--seed',seed);put('--cut-scenario',scenario)
            rows.append(dict(id=name,scenario=scenario,method='native',cmd=cmd,latent_frames=128,cohort_pair=lane,
                repeat_reason='new_same_prefix_native_state_feasibility',formal_holdout=False))
    return rows
