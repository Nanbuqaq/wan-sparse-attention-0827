"""Fixed mixed-bank reads and separate denoise-route reuse, balanced pairs."""
def build_read_route(spec,assets,source,output,seed,base_builder):
    if seed!=20261010:raise ValueError('frozen master registration uses existing seeds13/1010')
    base=base_builder(spec,'native',assets,source,output,spec['development_seed']);rows=[]
    groups=[('generated_bead_state_cut_revisit',20260913,0,'keep'),
            ('generated_bead_state_cut_revisit',20260913,1,'update'),
            ('w2_rotating_wooden_bird',20261010,0,None),('w2_tracking_delivery_cart',20261010,1,None)]
    for task,case_seed,lane,request in groups:
        template=next((r for r in base if r['scenario']==task),base[0])
        variants=['native','all','old','new'] if request else ['native','every_step','first_only','dual_02']
        for variant in variants:
            label=f'{request}_{variant}' if request else variant
            name=f'{task}__s{case_seed}__{label}';cmd=list(template['cmd'])
            def put(key,value):
                if key in cmd:cmd[cmd.index(key)+1]=str(value)
                else:cmd.extend([key,str(value)])
            put('--output',output/name);put('--cut-scenario',task);put('--seed',case_seed)
            if request:
                put('--request-compatibility-fork',request)
                if variant!='native':
                    put('--wave2-method','w2_full_recall');put('--wave2-version-policy','old4_new4');put('--version-read',variant)
            elif variant!='native':
                put('--wave2-method','w2_steady_sparse');put('--wave2-selector','query_sum_batch4');put('--wave2-preparation','geometry_cache')
                put('--wave2-route-refresh',variant);cmd+=['--wave2-age-observer']
            rows.append(dict(id=name,scenario=task,method=label,cmd=cmd,latent_frames=128,cohort_pair=lane,
                repeat_reason='fixed_request_version_read_or_denoise_route_reuse_with_own_controls',formal_holdout=False))
    return rows
