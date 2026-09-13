"""Two fixed-seed multi-event capacity controls; no future source table online."""
def build_multi_event(spec,assets,source,output,seed,base_builder,*,lineage=False):
    if seed!=20260913:raise ValueError('multi-event registration uses seeds13/14')
    template=base_builder(spec,'native',assets,source,output,spec['development_seed'])[0];rows=[]
    for lane,case_seed in enumerate((20260913,20260914)):
        methods=('raw_latest','canonical_latest','raw_max','canonical_max','canonical_max6') if lineage else ('native','release','restore8','restore6')
        for method in methods:
            name=f'w2_multi_event__s{case_seed}__{method}';cmd=list(template['cmd'])
            def put(key,value):
                if key in cmd:cmd[cmd.index(key)+1]=str(value)
                else:cmd.extend([key,str(value)])
            put('--output',output/name);put('--cut-scenario','w2_multi_event');put('--seed',case_seed)
            if lineage:
                put('--wave2-method','w2_scene_release');put('--scene-access-mode','restore_anchor');cmd+=['--scene-payload-catalog']
                if method.startswith('canonical'):cmd+=['--scene-canonical-identity']
                if 'max' in method:put('--scene-ranking','max_similarity')
                if method.endswith('6'):put('--scene-archive-gib',6)
            elif method!='native':
                put('--wave2-method','w2_scene_release');put('--scene-access-mode','release_broad' if method=='release' else 'restore_anchor')
            if not lineage and method.startswith('restore'):
                put('--scene-archive-gib',method[-1]);cmd+=['--scene-payload-catalog']
            rows.append(dict(id=name,scenario='w2_multi_event',method=method,cmd=cmd,latent_frames=128,cohort_pair=lane,
                repeat_reason='new_multi_event_and_capacity_miss_diagnostic',formal_holdout=False))
    return rows
