"""Fixed two-read hypotheses on state and generated-detail contexts."""


def build_source_two(spec,assets,source,output,seed,base_builder):
    if seed!=20261023:raise ValueError('two-read wave uses frozen per-context seeds23/30/31')
    template=base_builder(spec,'source_clean',assets,source,output,20261023)[0]
    contexts=(('w2_state_red_toolbox_keep',20261023),('w2_state_red_toolbox_close',20261023),
              ('w2_state_pattern_tile_keep',20261030),('w2_state_pattern_tile_keep',20261031))
    rows=[]
    for lane,(task,case_seed) in enumerate(contexts):
        for policy in ('all','first_last','first_clean','last_clean'):
            row=dict(template,cmd=list(template['cmd']))
            name=f'{task}__s{case_seed}__source_two_{policy}'
            for k,v in [('--output',str(output/name)),('--cut-scenario',task),('--seed',str(case_seed)),
                        ('--source-snapshot-window','latest8'),('--source-stage-policy',policy)]:
                row['cmd'][row['cmd'].index(k)+1]=v
            row.update(id=name,scenario=task,method=policy,source_window='latest8',cohort_pair=lane,
                repeat_reason='fixed_two_read_anchor_refinement_vs_equal_no_first_control')
            rows.append(row)
    return rows
