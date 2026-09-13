"""Frozen homogeneous cohorts; complete task groups stay on one physical pair."""
from scripts.next24h_cohort import build_cohort


def build_access_cohort(spec,stage,assets,source,output,seed,base_builder):
    if seed!=20261010:raise ValueError('first access/motion cohort uses existing development seed')
    base=base_builder(spec,'native',assets,source,output,seed)
    tasks=(['w2_ceramic_jug_revisit','w2_settled_pebble_bowl'] if stage=='access_factorial' else
           ['w2_ceramic_jug_revisit','w2_settled_pebble_bowl','w2_rotating_wooden_bird','w2_tracking_delivery_cart'])
    groups=[]
    for lane,task in enumerate(tasks):
        template=next(r for r in base if r['scenario']==task)
        variants=(['native','release_broad','release_narrow','restore_broad','restore_narrow'] if stage=='access_factorial' else
                  ['keep_copy','no_copy'] if lane<2 else
                  ['native','sum_fast','recent','bridge','early_heavy','late_heavy'])
        # Keep correctness reference before no_copy; otherwise reverse independent
        # method ordering between the two complementary tasks.
        if lane==3:variants=list(reversed(variants))
        rows=[]
        for variant in variants:
            name=f'{task}__s{seed}__{variant}';cmd=list(template['cmd'])
            def put(key,value):
                if key in cmd:cmd[cmd.index(key)+1]=str(value)
                else:cmd.extend([key,str(value)])
            put('--output',output/name)
            if variant in ('keep_copy','no_copy','release_broad','release_narrow','restore_broad','restore_narrow'):
                put('--wave2-method','w2_scene_release')
                if variant=='no_copy':
                    cmd+=['--scene-no-retired-copy','--equivalence-reference',str(output/f'{task}__s{seed}__keep_copy/summary.json')]
                elif variant!='keep_copy':put('--scene-access-mode',variant)
            elif variant!='native':
                put('--wave2-method','w2_steady_sparse');put('--wave2-preparation','geometry_cache')
                put('--wave2-selector','query_sum_batch4' if variant=='sum_fast' else 'recent_bridge' if variant=='bridge' else 'recent_no_score')
                if variant in ('early_heavy','late_heavy'):put('--wave2-stage-budget',variant)
            rows.append(dict(id=name,scenario=task,method=variant,cmd=cmd,latent_frames=128,
                cohort_pair=lane,repeat_reason='new_intervention_or_matched_strong_control',formal_holdout=False))
        groups.append(rows)
    result=[group[i] for i in range(max(map(len,groups))) for group in groups if i<len(group)]
    for row in result:
        if row['method']=='no_copy':
            row['reference_case_index']=next(i for i,x in enumerate(result) if x['scenario']==row['scenario'] and x['method']=='keep_copy')
    return result
