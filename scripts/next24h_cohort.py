"""Complete same-allocation native controls and fixed-protocol recall regressions."""
from pathlib import Path


def build_cohort(spec,stage,assets,source,output,seed,base_builder):
    if stage=='long_quality_replication':
        if seed!=spec['replication_seed']:raise ValueError('long quality replication uses reserved20261011 seed')
        base=build_cohort(spec,'matched_controls',assets,source,output,spec['development_seed'],base_builder)
        rows=[r for r in base if r['scenario']=='w2_rotating_wooden_bird' and r['method'] in ('native','sum_fast')]
        for row in rows:
            row['id']=f"{row['scenario']}__s{seed}__{row['method']}__latent728"
            row['cmd'][row['cmd'].index('--output')+1]=str(output/row['id']);row['cmd'][row['cmd'].index('--seed')+1]=str(seed)
            row['cmd']+=['--duration-probe-latents','728'];row.update(latent_frames=728,cohort_pair=0,repeat_reason='single_reserved_seed_quality_signal_check')
        return rows
    if stage=='long_sum_regression':
        if seed!=20261010:raise ValueError('long regression freezes seed20261010')
        base=build_cohort(spec,'matched_controls',assets,source,output,seed,base_builder)
        chosen=[r for r in base if r['scenario']=='w2_rotating_wooden_bird' and r['method'] in ('native','sum_old','sum_fast')]
        for row in chosen:
            row['id']+='__latent728';row['cmd'][row['cmd'].index('--output')+1]=str(output/row['id'])
            row['cmd']+=['--duration-probe-latents','728'];row.update(latent_frames=728,repeat_reason='long_memory_lifecycle_regression',cohort_pair=0)
        fast=next(r for r in chosen if r['method']=='sum_fast');old=next(r for r in chosen if r['method']=='sum_old')
        fast['cmd']+=['--equivalence-reference',str(output/old['id']/'summary.json')];fast['reference_case_index']=chosen.index(old)
        return chosen
    if stage=='recall_replication':
        if seed!=20260914:raise ValueError('registered recall replication is seed20260914')
        toy=build_cohort(spec,'recall_toy',assets,source,output,seed,base_builder)
        bead=list(reversed(build_cohort(spec,'recall_bead',assets,source,output,seed,base_builder)))
        result=[]
        for i in range(4):
            for lane,group in enumerate((toy,bead)):
                row=group[i];row.update(cohort_pair=lane,repeat_reason='registered_second_seed_regression',formal_holdout=False)
                result.append(row)
        return result
    if stage=='shared_route_holdout':
        if seed!=20261025:raise ValueError('shared-route holdout uses the frozen state-conversion seed')
        import sys
        runner=Path(__file__).resolve().parent/'run_longlive2_native_reference.py'
        cases=[]
        for label in ('native','shared25'):
            name=f'w2_state_red_toolbox_last_open__s{seed}__{label}'
            cmd=[sys.executable,str(runner),'--assets',str(assets),'--source',str(source),'--output',str(output/name),
                '--cut-scenario','w2_state_red_toolbox_last_open','--seed',str(seed),'--wave2-method','w2_native',
                '--native-local-frames','32','--cfg1-positive-cache-only','--native-inplace-cache','--native-shared-conditioning',
                '--fixed-adaln-warps','16','--fixed-adaln-stages','1','--constructor-mode','strict_checkpoint_no_parameter_init',
                '--pipeline-mode','overlap','--pipeline-encode-mode','thread','--wave2-steady-fraction','.5']
            def put(key,value):
                if key in cmd:cmd[cmd.index(key)+1]=str(value)
                else:cmd.extend([key,str(value)])
            put('--output',output/name);put('--seed',seed)
            if label=='shared25':
                put('--wave2-method','w2_steady_sparse');put('--wave2-selector','shared_sum_block64')
                put('--wave2-preparation','geometry_cache');put('--wave2-steady-fraction','.25')
            cases.append(dict(id=name,scenario='w2_state_red_toolbox_last_open',method=label,cmd=cmd,
                latent_frames=spec['latent_frames'],repeat=0,cohort_pair=0,
                repeat_reason='state_transition_quality_holdout',not_independent_quality_sample=False))
        return cases
    if stage=='shared_route_repeats':
        if seed!=20261010:raise ValueError('shared-route candidate uses the frozen performance seed')
        base=base_builder(spec,'native',assets,source,output,spec['development_seed'])
        tasks=['w2_rotating_wooden_bird','w2_tracking_delivery_cart']
        cases=[]
        for task_index,task in enumerate(tasks):
            template=next(c for c in base if c['scenario']==task)
            for repeat in range(3):
                order=('shared25','native') if (repeat+task_index)%2==0 else ('native','shared25')
                for label in order:
                    name=f'{task}__s{seed}__{label}_r{repeat}';cmd=list(template['cmd'])
                    def put(key,value):
                        if key in cmd:cmd[cmd.index(key)+1]=str(value)
                        else:cmd.extend([key,str(value)])
                    put('--output',output/name);put('--seed',seed);put('--cut-scenario',task)
                    if label=='shared25':
                        put('--wave2-method','w2_steady_sparse');put('--wave2-selector','shared_sum_block64')
                        put('--wave2-preparation','geometry_cache');put('--wave2-steady-fraction','.25')
                    else:
                        put('--wave2-method','w2_native')
                    cases.append(dict(id=name,scenario=task,method=label,cmd=cmd,latent_frames=128,
                        repeat=repeat,repeat_reason='paired_shared_block64_route_candidate',cohort_pair=task_index,
                        not_independent_quality_sample=True))
        return cases
    continuous=stage in ('matched_controls','timing_repeats','scene_release','recent_control','recent_hopper_control')
    if continuous and seed!=20261010:raise ValueError('matched controls freeze development seed')
    if stage.startswith('recall_') and seed not in (20260913,20260914):raise ValueError('recall regression freezes seeds20260913/14')
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
        if stage=='recent_control':variants=['native','recent_no_score'] if task_index==0 else ['recent_no_score','native']
        if stage=='recent_hopper_control':variants=['native','steady_mass50','sum_fast','recent_no_score'] if task_index==0 else ['recent_no_score','sum_fast','steady_mass50','native']
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
            if variant=='recent_no_score':put('--wave2-selector','recent_no_score')
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
