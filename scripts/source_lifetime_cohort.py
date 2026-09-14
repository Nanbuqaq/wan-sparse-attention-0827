"""Native plus four same-reader layer/lifetime arms on two fixed protocols."""


def build_source_lifetime(spec,assets,source,output,seed,base_builder):
    if seed!=20261010:raise ValueError('source lifetime cohort uses frozen development seeds10/13')
    base=base_builder(spec,'native',assets,source,output,seed);rows=[]
    for lane,(task,case_seed,motion) in enumerate((('w2_ceramic_jug_revisit',20261010,False),
                                                  ('generated_patchwork_toy_cut_revisit',20260913,True))):
        template=next((x for x in base if x['scenario']==task),base[0])
        for policy in ('native','full_once','prior_once','full_three','prior_three'):
            name=f'{task}__s{case_seed}__lifetime_{policy}'+('_motion' if motion else '')
            cmd=list(template['cmd'])
            def put(key,value):
                if key in cmd:cmd[cmd.index(key)+1]=str(value)
                else:cmd.extend([key,str(value)])
            put('--output',output/name);put('--seed',case_seed);put('--cut-scenario',task)
            cmd+=['--source-lifetime-study']
            if motion:cmd+=['--source-lifetime-motion']
            if policy!='native':
                put('--wave2-method','w2_full_recall');put('--source-lifetime-policy',policy)
                put('--source-lifetime-backend','concat')
            rows.append(dict(id=name,scenario=task,method=policy,cmd=cmd,latent_frames=128,
                cohort_pair=lane,repeat_reason='new_immutable_side_reader_layer_lifetime_controls',formal_holdout=False))
    return rows
