"""Registered long motion diagnosis and reversed matched archive timings."""
def build_followup(spec,stage,assets,source,output,seed,base_builder):
    if seed!=(20260913 if stage=='delayed_and_return' else 20261010):raise ValueError('followups retain registered development/replication seeds')
    base=base_builder(spec,'native',assets,source,output,spec['development_seed']);rows=[]
    groups=([(task,seed,['native','restore_broad','beta1','beta_half','beta2']) for task in ('w2_ceramic_jug_revisit','w2_settled_pebble_bowl')] if stage=='source_weight' else
            [('generated_patchwork_toy_cut_revisit',seed,['native','release_broad','release_narrow','restore_broad','restore_narrow']),
             ('generated_bead_state_cut_revisit',seed,['native','recent','early_heavy','late_heavy'])] if stage=='delayed_and_return' else
            [('w2_rotating_wooden_bird',20261010,['native','sum_fast','recent','bridge']),
             ('w2_rotating_wooden_bird',20261011,['bridge','recent','sum_fast','native'])] if stage=='motion_long' else
            [(task,seed,['no_copy_r0','keep_copy_r0','keep_copy_r1','no_copy_r1']) for task in ('w2_ceramic_jug_revisit','w2_settled_pebble_bowl')])
    for lane,(task,case_seed,variants) in enumerate(groups):
        template=next((r for r in base if r['scenario']==task),base[0])
        for variant in variants:
            name=f'{task}__s{case_seed}__{variant}'+('__latent728' if stage=='motion_long' else '')
            cmd=list(template['cmd'])
            def put(key,value):
                if key in cmd:cmd[cmd.index(key)+1]=str(value)
                else:cmd.extend([key,str(value)])
            put('--output',output/name);put('--seed',case_seed);put('--cut-scenario',task)
            if stage in ('source_weight','delayed_and_return'):
                if stage=='source_weight' and variant!='native':
                    put('--wave2-method','w2_scene_release');put('--scene-access-mode','restore_broad')
                    if variant!='restore_broad':put('--source-memory-beta',{'beta1':1.,'beta_half':.5,'beta2':2.}[variant])
                elif task=='generated_patchwork_toy_cut_revisit' and variant!='native':
                    put('--wave2-method','w2_scene_release');put('--scene-access-mode',variant)
                elif task=='generated_bead_state_cut_revisit' and variant!='native':
                    put('--wave2-method','w2_steady_sparse');put('--wave2-selector','recent_no_score')
                    if variant!='recent':put('--wave2-stage-budget',variant)
            elif stage=='motion_long':
                if variant!='native':
                    put('--wave2-method','w2_steady_sparse');put('--wave2-preparation','geometry_cache')
                    put('--wave2-selector',{'sum_fast':'query_sum_batch4','recent':'recent_no_score','bridge':'recent_bridge'}[variant])
                put('--duration-probe-latents',728)
            else:
                put('--wave2-method','w2_scene_release')
                if variant.startswith('no_copy'):cmd+=['--scene-no-retired-copy']
            rows.append(dict(id=name,scenario=task,method=variant,cmd=cmd,latent_frames=728 if stage=='motion_long' else 128,
                cohort_pair=lane,repeat_reason='new_memory_strength_or_delayed_source_and_return_stage_budget' if stage in ('source_weight','delayed_and_return') else 'matched_long_motion_controls' if stage=='motion_long' else 'reverse_order_timing_after_initial_warmup_confound',formal_holdout=False))
    return rows
