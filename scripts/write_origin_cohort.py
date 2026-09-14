"""Two bounded controls distinguish capacity coalescing and anchor fidelity."""


def build_write_origin(spec,assets,source,output,seed,base_builder):
    if seed!=20260913:raise ValueError('write pilot fixes the existing diagnostic seeds13/14')
    template=base_builder(spec,'native',assets,source,output,spec['development_seed'])[0];rows=[]
    for lane,(task,case_seed,gib) in enumerate((('w2_write_return_b',20260913,6),('w2_write_return_a',20260914,8))):
        for policy in ('native','append','root_latest_fifo','skip_derived'):
            name=f'{task}__s{case_seed}__{gib}GiB__{policy}';cmd=list(template['cmd'])
            def put(key,value):
                if key in cmd:cmd[cmd.index(key)+1]=str(value)
                else:cmd.extend([key,str(value)])
            put('--output',output/name);put('--cut-scenario',task);put('--seed',case_seed)
            if policy!='native':
                put('--wave2-method','w2_scene_release');put('--scene-access-mode','restore_anchor')
                put('--scene-archive-gib',gib);put('--scene-ranking','max_similarity');put('--scene-write-policy',policy)
                cmd+=['--scene-payload-catalog','--scene-canonical-identity']
            rows.append(dict(id=name,scenario=task,method=policy,cmd=cmd,latent_frames=128,cohort_pair=lane,
                repeat_reason='write_origin_vs_root_FIFO_capacity_control',formal_holdout=False))
    return rows
