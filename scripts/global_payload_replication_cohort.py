"""Independent state/motion validation of global matching/value roles."""


def build_global_payload_replication(spec,assets,source,output,seed,base_builder):
    if seed!=20261023:raise ValueError('global-role replication fixes toolbox seed23')
    native=base_builder(spec,'native',assets,source,output,spec['development_seed'])[0]
    raw=base_builder(spec,'source_two',assets,source,output,seed)[0]
    rows=[]
    for lane,(scenario,motion) in enumerate((('w2_state_red_toolbox_close',False),('w2_state_red_toolbox_keep',True))):
        for mode in ('native','original','zero_v','zero_kv','mean_v'):
            template=native if mode=='native' else raw;cmd=list(template['cmd'])
            if '--source-clean-cache-witness' in cmd:cmd.remove('--source-clean-cache-witness')
            name=f'{scenario}__s{seed}__global_role_{"motion_" if motion else ""}{mode}'
            for k,v in [('--output',str(output/name)),('--cut-scenario',scenario),('--seed',str(seed))]:cmd[cmd.index(k)+1]=v
            if motion:cmd+=['--state-quarter-turn']
            if mode!='native':cmd+=['--global-payload',mode,'--archive-write-backend','sync','--archive-readiness','device','--archive-skip-resident-global']
            rows.append(dict(template,id=name,scenario=scenario,method=mode,cmd=cmd,cohort_pair=lane,
                formal_holdout=False,repeat_reason='independent_state_and_motion_request_global_key_value_replication',
                common_archive_elision=mode!='native'))
    return rows
