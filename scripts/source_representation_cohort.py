"""Frozen actual-content versus contextual-representation comparison."""


def build_source_representation(spec,assets,source,output,seed,base_builder):
    if seed!=20261023:raise ValueError('representation wave fixes per-context seeds23/30/31')
    template=base_builder(spec,'source_two',assets,source,output,seed)[0]
    contexts=(('w2_state_red_toolbox_keep',20261023),('w2_state_red_toolbox_close',20261023),
              ('w2_state_pattern_tile_keep',20261030),('w2_state_pattern_tile_keep',20261031))
    rows=[]
    for lane,(task,case_seed) in enumerate(contexts):
        for representation in ('raw_record','past_reencode','current_reencode'):
            cmd=list(template['cmd']);cmd.remove('--source-clean-cache-witness')
            name=f'{task}__s{case_seed}__representation_{representation}'
            for k,v in [('--output',str(output/name)),('--cut-scenario',task),('--seed',str(case_seed))]:
                cmd[cmd.index(k)+1]=v
            cmd+=['--source-representation',representation,'--audit-shared-conditioning-inputs']
            rows.append(dict(template,id=name,scenario=task,method=representation,cohort_pair=lane,cmd=cmd,
                repeat_reason='new_source_representation_with_raw_recorder_equivalence_control',
                ordinary_source_read_budget_fixed=True,auxiliary_compute_is_additional=True,
                raw_archives_still_retained=True,formal_holdout=False))
    return rows
