"""Single source-call sufficiency and equal-call timing controls."""


def build_source_first(spec,assets,source,output,seed,base_builder):
    if seed!=20261023:raise ValueError('first-source pilot uses the frozen toolbox seed23')
    base=base_builder(spec,'source_clean',assets,source,output,seed)
    rows=[]
    for lane,task in enumerate(('w2_state_red_toolbox_keep','w2_state_red_toolbox_close')):
        template=next(x for x in base if x['scenario']==task and x['source_window']=='latest8' and x['method']=='all')
        for policy in ('all','first_only','second_only','last_only','clean_only'):
            row=dict(template,cmd=list(template['cmd']))
            name=f'{task}__s{seed}__source_once_{policy}'
            row['cmd'][row['cmd'].index('--output')+1]=str(output/name)
            row['cmd'][row['cmd'].index('--source-stage-policy')+1]=policy
            row.update(id=name,method=policy,cohort_pair=lane,
                repeat_reason='first_read_sufficiency_vs_other_equal_single_call_positions')
            rows.append(row)
    return rows
