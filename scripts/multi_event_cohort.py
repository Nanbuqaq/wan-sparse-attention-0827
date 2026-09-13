"""Two fixed-seed multi-event capacity controls; no future source table online."""
def build_multi_event(spec,assets,source,output,seed,base_builder):
    if seed!=20260913:raise ValueError('multi-event registration uses seeds13/14')
    template=base_builder(spec,'native',assets,source,output,spec['development_seed'])[0];rows=[]
    for lane,case_seed in enumerate((20260913,20260914)):
        for method in ('native','release','restore8','restore6'):
            name=f'w2_multi_event__s{case_seed}__{method}';cmd=list(template['cmd'])
            def put(key,value):
                if key in cmd:cmd[cmd.index(key)+1]=str(value)
                else:cmd.extend([key,str(value)])
            put('--output',output/name);put('--cut-scenario','w2_multi_event');put('--seed',case_seed)
            if method!='native':
                put('--wave2-method','w2_scene_release');put('--scene-access-mode','release_broad' if method=='release' else 'restore_anchor')
            if method.startswith('restore'):
                put('--scene-archive-gib',method[-1]);cmd+=['--scene-payload-catalog']
            rows.append(dict(id=name,scenario='w2_multi_event',method=method,cmd=cmd,latent_frames=128,cohort_pair=lane,
                repeat_reason='new_multi_event_and_capacity_miss_diagnostic',formal_holdout=False))
    return rows
