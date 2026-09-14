"""Fixed conditional-difference wave with paired native/raw/direction controls."""


def build_conditional_delta(spec,assets,source,output,seed,base_builder):
    if seed!=20261023:raise ValueError('conditional wave fixes seeds23/30/31')
    native=base_builder(spec,'native',assets,source,output,spec['development_seed'])[0]
    raw=base_builder(spec,'source_representation',assets,source,output,seed)[0]
    contexts=(('w2_state_red_toolbox_close',20261023,False),('w2_state_red_toolbox_keep',20261023,True),
              ('w2_state_pattern_tile_keep',20261030,False),('w2_state_pattern_tile_keep',20261031,False))
    rows=[]
    for lane,(task,case_seed,motion) in enumerate(contexts):
        methods=('native','raw_record','null','forward','reverse') if lane==0 else ('native','raw_record','forward','reverse')
        raw_id=None
        for method in methods:
            template=native if method=='native' else raw;cmd=list(template['cmd'])
            name=f'{task}__s{case_seed}__delta_{"motion_" if motion else ""}{method}'
            for key,value in [('--output',str(output/name)),('--cut-scenario',task),('--seed',str(case_seed))]:
                cmd[cmd.index(key)+1]=value
            if '--audit-shared-conditioning-inputs' not in cmd:cmd+=['--audit-shared-conditioning-inputs']
            if motion:cmd+=['--state-quarter-turn']
            if method in ('forward','reverse','null'):cmd+=['--source-conditional-delta',method]
            if method=='raw_record':raw_id=name
            if method=='null':cmd+=['--equivalence-reference',str(output/raw_id/'summary.json')]
            rows.append(dict(template,id=name,scenario=task,method=method,cohort_pair=lane,cmd=cmd,
                current_motion_counterexample=motion,formal_holdout=False,
                repeat_reason='matched_hardware_and_full_cost_controls_for_new_conditional_correction',
                opposite_directions_share_two_auxiliary_forwards=method in ('forward','reverse')))
            if method=='null':rows[-1]['reference_case_index']=len(rows)-2
    return rows
