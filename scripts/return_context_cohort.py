"""Keep source fixed while identifying stale pin versus recent-away context."""


def build_return_context(spec,assets,source,output,seed,base_builder):
    if seed!=20261010:raise ValueError('return-context pilot fixes existing development seeds10/13')
    base=base_builder(spec,'native',assets,source,output,seed);rows=[]
    for lane,(task,case_seed,motion) in enumerate((('w2_ceramic_jug_revisit',20261010,False),
                                                 ('generated_patchwork_toy_cut_revisit',20260913,True))):
        template=next((x for x in base if x['scenario']==task),base[0])
        for policy in ('native','legacy_slot','full','pin_first','recent_first','both_first','anchor_transition','anchor_no_source'):
            name=f'{task}__s{case_seed}__context_{policy}'+('_motion' if motion else '');cmd=list(template['cmd'])
            def put(key,value):
                if key in cmd:cmd[cmd.index(key)+1]=str(value)
                else:cmd.extend([key,str(value)])
            put('--output',output/name);put('--cut-scenario',task);put('--seed',case_seed)
            cmd+=['--source-lifetime-study','--return-context-study']
            if motion:cmd+=['--source-lifetime-motion']
            if policy!='native':put('--wave2-method','w2_full_recall')
            if policy not in ('native','legacy_slot'):
                put('--source-lifetime-policy','off' if policy=='anchor_no_source' else 'full_once')
                put('--source-context-policy','anchor_transition' if policy=='anchor_no_source' else policy)
                put('--source-packing-order','after_global')
            rows.append(dict(id=name,scenario=task,method=policy,cmd=cmd,latent_frames=128,cohort_pair=lane,
                repeat_reason='separate_recalled_source_from_stale_native_pin_and_recent_context',formal_holdout=False))
    return rows
