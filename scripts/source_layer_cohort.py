"""Fixed source-layer priors on identity return and explicit state override."""
def build_source_layers(spec,assets,source,output,seed,base_builder):
    if seed!=20261010:raise ValueError('source-layer master registration fixes existing seeds10/13')
    base=base_builder(spec,'native',assets,source,output,spec['development_seed']);rows=[]
    for lane,(task,case_seed,override) in enumerate((('w2_ceramic_jug_revisit',20261010,False),('generated_bead_state_cut_revisit',20260913,True))):
        template=next((x for x in base if x['scenario']==task),base[0])
        for policy in ('native','full','uniform_third','prior10','early10','late10'):
            label=('update_' if override else '')+policy;name=f'{task}__s{case_seed}__layers_{label}';cmd=list(template['cmd'])
            def put(key,value):
                if key in cmd:cmd[cmd.index(key)+1]=str(value)
                else:cmd.extend([key,str(value)])
            put('--output',output/name);put('--seed',case_seed);put('--cut-scenario',task)
            if override:put('--request-compatibility-fork','update')
            if policy!='native':
                put('--wave2-method','w2_scene_release');put('--scene-access-mode','restore_broad');put('--source-layer-policy',policy)
            rows.append(dict(id=name,scenario=task,method=label,cmd=cmd,latent_frames=128,cohort_pair=lane,
                repeat_reason='equal_source_frame_layer_exposure_with_published_prior_control',formal_holdout=False))
    return rows
