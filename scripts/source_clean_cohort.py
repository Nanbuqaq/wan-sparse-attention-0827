"""Read during denoising versus committing future KV; matched four-call controls."""


def build_source_clean(spec,assets,source,output,seed,base_builder):
    if seed!=20261023:raise ValueError('source-clean cohort freezes reviewed seeds22/23')
    template=base_builder(spec,'native',assets,source,output,spec['development_seed'])[0]
    contexts=(('w2_state_red_toolbox_keep',20261023,'oldest_resident8'),
              ('w2_state_red_toolbox_close',20261023,'latest8'),
              ('w2_state_silver_case_close',20261022,'oldest_resident8'),
              ('w2_state_red_toolbox_keep',20261023,'latest8'))
    rows=[]
    for lane,(task,case_seed,window) in enumerate(contexts):
        ref=None
        for policy in ('all','no_clean','no_first','no_last'):
            row=dict(template,cmd=list(template['cmd']))
            name=f'{task}__s{case_seed}__clean_{window}_{policy}'
            def put(k,v):
                if k in row['cmd']:row['cmd'][row['cmd'].index(k)+1]=str(v)
                else:row['cmd']+=[k,str(v)]
            for k,v in [('--output',output/name),('--seed',case_seed),('--cut-scenario',task),('--wave2-method','w2_full_recall'),
                ('--source-lifetime-policy','full_once'),('--source-context-policy','anchor_transition'),('--source-packing-order','after_global'),
                ('--source-snapshot-window',window),('--source-stage-policy',policy)]:put(k,v)
            row['cmd']+=['--source-lifetime-study','--return-context-study','--source-clean-cache-witness']
            row.update(id=name,scenario=task,method=policy,source_window=window,cohort_pair=lane,latent_frames=128,
                repeat_reason='new_matched_source_read_vs_clean_writeback_factor',formal_holdout=False)
            if policy=='all':ref=(len(rows),name)
            if policy=='no_clean':
                row['reference_case_index']=ref[0]
                put('--source-prefix-reference',output/ref[1]/'summary.json')
            rows.append(row)
    return rows
