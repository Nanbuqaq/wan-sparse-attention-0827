"""State requests on latest versus actually resident older source, same raw bytes."""


def build_state_snapshot(spec,assets,source,output,seed,base_builder):
    if seed!=20261021:raise ValueError('state snapshot pilot fixes the reviewed silver-case development seed')
    template=base_builder(spec,'native',assets,source,output,spec['development_seed'])[0];rows=[]
    for lane,operation in enumerate(('keep','close')):
        scenario=f'w2_state_silver_case_{operation}'
        for method in ('native','anchor_no_source','latest8','oldest_resident8'):
            name=f'{scenario}__s{seed}__snapshot_{method}';cmd=list(template['cmd'])
            def put(key,value):
                if key in cmd:cmd[cmd.index(key)+1]=str(value)
                else:cmd.extend([key,str(value)])
            put('--output',output/name);put('--cut-scenario',scenario);put('--seed',seed)
            if method!='native':
                put('--wave2-method','w2_full_recall');cmd+=['--source-lifetime-study','--return-context-study']
                put('--source-lifetime-policy','off' if method=='anchor_no_source' else 'full_once')
                put('--source-context-policy','anchor_transition');put('--source-packing-order','after_global')
                put('--source-snapshot-window','oldest_resident8' if method=='oldest_resident8' else 'latest8')
            rows.append(dict(id=name,scenario=scenario,method=method,cmd=cmd,latent_frames=128,cohort_pair=lane,
                repeat_reason='observed_source_window_validity_before_intent_admission',formal_holdout=False))
        # Independent native feasibility only; do not promote a new-seed memory
        # method before checking that seed's generated source and state request.
        extra=dict(rows[-4]);extra['cmd']=list(extra['cmd']);extra_seed=20261022
        extra_name=f'{scenario}__s{extra_seed}__native_feasibility'
        extra['cmd'][extra['cmd'].index('--seed')+1]=str(extra_seed)
        extra['cmd'][extra['cmd'].index('--output')+1]=str(output/extra_name)
        extra.update(id=extra_name,method='native_second_seed',repeat_reason='independent_native_source_and_state_feasibility')
        rows.append(extra)
    return rows
