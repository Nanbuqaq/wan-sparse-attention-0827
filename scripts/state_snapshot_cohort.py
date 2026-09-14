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


def build_state_snapshot_replication(spec,assets,source,output,seed,base_builder):
    """Second-seed mechanism replication plus an unseen protocol feasibility pair."""
    if seed!=20261022:raise ValueError('snapshot replication uses reserved seed22; new toolbox uses23')
    pilot=build_state_snapshot(spec,assets,source,output,20261021,base_builder)
    rows=[]
    for old in pilot:
        if old['method']=='native_second_seed':continue
        row=dict(old,cmd=list(old['cmd']))
        row['id']=old['id'].replace('s20261021__snapshot','s20261022__snapshot_replication')
        row['cmd'][row['cmd'].index('--seed')+1]=str(seed)
        row['cmd'][row['cmd'].index('--output')+1]=str(output/row['id'])
        row['repeat_reason']=('new_same_physical_pair_native_control_for_replication_cost' if row['method']=='native'
                              else 'independent_seed_of_frozen_source_window_and_request_interaction')
        rows.append(row)
    for lane,operation in enumerate(('keep','close')):
        template=next(x for x in rows if x['method']=='native' and x['cohort_pair']==lane)
        row=dict(template,cmd=list(template['cmd']))
        row['scenario']=f'w2_state_red_toolbox_{operation}'
        row['id']=f"{row['scenario']}__s20261023__native_feasibility"
        for key,value in [('--cut-scenario',row['scenario']),('--seed','20261023'),('--output',str(output/row['id']))]:
            row['cmd'][row['cmd'].index(key)+1]=value
        row['method']='native_new_protocol'
        row['repeat_reason']='unseen_protocol_native_source_and_request_feasibility_before_memory_promotion'
        rows.append(row)
    return rows
