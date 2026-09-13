"""Complete same-allocation native controls and fixed-protocol recall regressions."""
from pathlib import Path


def build_cohort(spec,stage,assets,source,output,seed,base_builder):
    continuous=stage in ('matched_controls','timing_repeats','scene_release')
    if continuous and seed!=20261010:raise ValueError('matched controls freeze development seed')
    if stage.startswith('recall_') and seed!=20260913:raise ValueError('first recall regression freezes seed20260913')
    base=base_builder(spec,'native',assets,source,output,spec['development_seed'])
    tasks=['w2_rotating_wooden_bird','w2_tracking_delivery_cart'] if continuous else [
        'generated_patchwork_toy_cut_revisit' if stage=='recall_toy' else 'generated_bead_state_cut_revisit']
    if stage=='scene_release':tasks=['w2_ceramic_jug_revisit','w2_settled_pebble_bowl']
    per_task=[]
    for task_index,task in enumerate(tasks):
        template=next(c for c in base if c['scenario']==tasks[task_index]) if continuous else base[0]
        variants=(['native','steady_mass50','sum_old','sum_fast'] if task_index==0 else ['sum_fast','sum_old','steady_mass50','native']) if stage=='matched_controls' else ['native','steady_mass50','full_recall','steady_plus_recall']
        if stage=='matched_controls':variants+=['sum_observer']
        if stage=='scene_release':variants=['native','scene_release']
        if stage=='timing_repeats':
            variants=[f'{v}_r{rep}' for rep in range(3) for v in
                ((['sum_old','sum_fast'] if rep%2==task_index%2 else ['sum_fast','sum_old'])+['native'])]
        items=[]
        for label in variants:
            variant=label.rsplit('_r',1)[0] if stage=='timing_repeats' else label
            name=f'{task}__s{seed}__{label}';cmd=list(template['cmd'])
            def put(key,value):
                if key in cmd:cmd[cmd.index(key)+1]=str(value)
                else:cmd.extend([key,str(value)])
            put('--output',output/name);put('--seed',seed);put('--cut-scenario',task)
            method={'native':'w2_native','full_recall':'w2_full_recall','steady_plus_recall':'w2_steady_plus_recall','scene_release':'w2_scene_release'}.get(variant,'w2_steady_sparse')
            put('--wave2-method',method)
            if variant.startswith('sum_'):
                put('--wave2-selector','query_sum_batch4');cmd+=['--wave2-route-audit']
                if variant in ('sum_fast','sum_observer'):put('--wave2-preparation','geometry_cache')
                if variant=='sum_observer':
                    cmd+=['--wave2-steady-observer'];put('--equivalence-reference',output/f'{task}__s{seed}__sum_fast/summary.json')
            items.append(dict(id=name,scenario=task,method=variant,cmd=cmd,latent_frames=128,
                repeat_reason='scene_control_diagnostic' if variant=='scene_release' else 'timing_replication' if stage=='timing_repeats' else 'observer_equivalence' if variant=='sum_observer' else 'same_route_optimization' if variant=='sum_fast' else 'new_homogeneous_control',
                not_independent_quality_sample=True,cohort_pair=task_index))
        per_task.append(items)
    cases=[items[i] for i in range(len(per_task[0])) for items in per_task]
    for i,c in enumerate(cases):
        if c['method']=='sum_observer':
            c['reference_case_index']=next(j for j,r in enumerate(cases) if r['scenario']==c['scenario'] and r['method']=='sum_fast')
            assert c['reference_case_index']<i
    return cases
