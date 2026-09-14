"""Past-text and raw-KV controls on a separately qualified object/state pair."""


def build_state_representation(spec,assets,source,output,seed,base_builder):
    if seed!=20261023:raise ValueError('representation pilot uses the qualified toolbox seed23')
    template=base_builder(spec,'native',assets,source,output,spec['development_seed'])[0]
    rows=[]
    for lane,operation in enumerate(('keep','close')):
        scenario=f'w2_state_red_toolbox_{operation}'
        for method in ('native','filter_no_archive','past_text_native','past_text_filter','raw_latest8','raw_oldest8'):
            row=dict(template,cmd=list(template['cmd']))
            name=f'{scenario}__s{seed}__representation_{method}'
            def put(k,v):
                if k in row['cmd']:row['cmd'][row['cmd'].index(k)+1]=str(v)
                else:row['cmd']+=[k,str(v)]
            put('--cut-scenario',scenario);put('--seed',seed);put('--output',output/name)
            if method not in ('native','past_text_native'):
                put('--wave2-method','w2_full_recall')
                row['cmd']+=['--source-lifetime-study','--return-context-study']
                put('--source-context-policy','anchor_transition');put('--source-packing-order','after_global')
                put('--source-lifetime-policy','full_once' if method.startswith('raw_') else 'off')
                put('--source-snapshot-window','oldest_resident8' if method=='raw_oldest8' else 'latest8')
                if not method.startswith('raw_'):row['cmd']+=['--source-no-archive']
            if method.startswith('past_text'):row['cmd']+=['--state-past-appearance-text']
            row.update(id=name,scenario=scenario,method=method,cohort_pair=lane,latent_frames=128,
                repeat_reason='new_same_pair_past_text_context_and_raw_source_comparison',formal_holdout=False,
                text_control_is_privileged_past_prompt_span=method.startswith('past_text'))
            rows.append(row)
    return rows
