"""Counterbalanced validation after independent elision/staging results."""


def build_combined_archive(spec,assets,source,output,seed,base_builder):
    if seed!=20261023:raise ValueError('combined archive wave fixes seeds23/30')
    native=base_builder(spec,'native',assets,source,output,spec['development_seed'])[0]
    raw=base_builder(spec,'source_two',assets,source,output,seed)[0]
    sequence=('native_start','dedup_r0','async_all_r0','combined_r0','combined_r1','async_all_r1','dedup_r1','native_end')
    rows=[]
    for lane,(task,case_seed,motion) in enumerate((('w2_state_red_toolbox_keep',20261023,True),('w2_state_pattern_tile_keep',20261030,False))):
        native_ref=raw_ref=None
        for variant in sequence:
            is_native=variant.startswith('native');template=native if is_native else raw;cmd=list(template['cmd'])
            if '--source-clean-cache-witness' in cmd:cmd.remove('--source-clean-cache-witness')
            name=f'{task}__s{case_seed}__archive_combined_{"motion_" if motion else ""}{variant}'
            for key,value in [('--output',str(output/name)),('--cut-scenario',task),('--seed',str(case_seed))]:cmd[cmd.index(key)+1]=value
            if motion:cmd+=['--state-quarter-turn']
            if not is_native:
                backend='sync' if variant.startswith('dedup') else 'staging'
                cmd+=['--archive-write-backend',backend,'--archive-readiness','device' if backend=='sync' else 'generation']
                if not variant.startswith('async_all'):cmd+=['--archive-skip-resident-global']
            reference=native_ref if is_native else raw_ref
            if reference:cmd+=['--equivalence-reference',str(output/reference[1]/'summary.json')]
            row=dict(template,id=name,scenario=task,method=variant,cohort_pair=lane,cmd=cmd,formal_holdout=False,
                repeat_reason='matched_combination_after_independent_elision_and_staging_evidence')
            if reference:row['reference_case_index']=reference[0]
            rows.append(row)
            if variant=='native_start':native_ref=(len(rows)-1,name)
            if variant=='dedup_r0':raw_ref=(len(rows)-1,name)
    return rows
