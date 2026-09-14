"""Matched archive-engine, overlap and readiness-scope timing controls."""


def build_archive_system(spec,assets,source,output,seed,base_builder):
    if seed!=20261023:raise ValueError('archive system wave fixes seeds23/30')
    native=base_builder(spec,'native',assets,source,output,spec['development_seed'])[0]
    raw=base_builder(spec,'source_two',assets,source,output,seed)[0]
    sequence=('native_start','sync_r0','serial_r0','async_device_r0','async_stream_r0',
              'async_stream_r1','async_device_r1','serial_r1','sync_r1','native_end')
    contexts=(('w2_state_red_toolbox_keep',20261023,True),('w2_state_pattern_tile_keep',20261030,False))
    rows=[]
    for lane,(task,case_seed,motion) in enumerate(contexts):
        native_ref=raw_ref=None
        for variant in sequence:
            is_native=variant.startswith('native');template=native if is_native else raw
            cmd=list(template['cmd'])
            if '--source-clean-cache-witness' in cmd:cmd.remove('--source-clean-cache-witness')
            name=f'{task}__s{case_seed}__archive_{"motion_" if motion else ""}{variant}'
            for k,v in [('--output',str(output/name)),('--cut-scenario',task),('--seed',str(case_seed))]:
                cmd[cmd.index(k)+1]=v
            if motion:cmd+=['--state-quarter-turn']
            backend=readiness=None
            if not is_native:
                backend='sync' if variant.startswith('sync_') else 'staging_serial' if variant.startswith('serial_') else 'staging'
                readiness='generation' if variant.startswith('async_stream_') else 'device'
                cmd+=['--archive-write-backend',backend,'--archive-readiness',readiness]
            reference=native_ref if is_native else raw_ref
            if reference is not None:cmd+=['--equivalence-reference',str(output/reference[1]/'summary.json')]
            row=dict(template,id=name,scenario=task,method=variant,cohort_pair=lane,cmd=cmd,
                archive_backend=backend,source_readiness_scope=readiness,formal_holdout=False,
                repeat_reason='same_hardware_counterbalanced_archive_engine_overlap_and_barrier_attribution',
                diagnostic_archive_hashes_disabled=True)
            if reference is not None:row['reference_case_index']=reference[0]
            rows.append(row)
            if variant=='native_start':native_ref=(len(rows)-1,name)
            if variant=='sync_r0':raw_ref=(len(rows)-1,name)
    return rows
