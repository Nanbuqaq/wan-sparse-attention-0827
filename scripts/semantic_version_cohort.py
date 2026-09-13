"""One frozen source/request/version wave plus two novelty tasks, own controls."""
def build_semantic_cohort(spec,assets,source,output,seed,base_builder):
    if seed!=20260913:raise ValueError('semantic/version pilot fixes existing bead development seed13')
    base=base_builder(spec,'native',assets,source,output,spec['development_seed'])
    groups=[('generated_bead_state_cut_revisit',0,['keep_native','keep_latest8','keep_old4_new4','keep_uniform8','update_native','update_latest8']),
            ('w2_rotating_wooden_bird',1,['native','sum_fast','value_novelty']),
            ('w2_tracking_delivery_cart',1,['value_novelty','sum_fast','native'])]
    rows=[]
    for task,lane,variants in groups:
        template=next((r for r in base if r['scenario']==task),base[0])
        for variant in variants:
            name=f'{task}__s{seed}__{variant}';cmd=list(template['cmd'])
            def put(key,value):
                if key in cmd:cmd[cmd.index(key)+1]=str(value)
                else:cmd.extend([key,str(value)])
            put('--output',output/name);put('--cut-scenario',task);put('--seed',seed)
            if task=='generated_bead_state_cut_revisit':
                request,policy=variant.split('_',1);put('--request-compatibility-fork',request)
                if policy!='native':put('--wave2-method','w2_full_recall');put('--wave2-version-policy',policy)
            elif variant!='native':
                put('--wave2-method','w2_steady_sparse');put('--wave2-preparation','geometry_cache')
                put('--wave2-selector','query_sum_batch4' if variant=='sum_fast' else 'value_novelty')
            rows.append(dict(id=name,scenario=task,method=variant,cmd=cmd,latent_frames=128,cohort_pair=lane,
                repeat_reason='new_source_request_version_or_novelty_intervention_with_own_native',formal_holdout=False))
    return rows
