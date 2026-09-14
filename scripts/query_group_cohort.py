"""Two complete task pairs separate granularity, group execution and selection."""


def build_query_groups(spec, assets, source, output, seed, base_builder):
    if seed != 20261010:
        raise ValueError('first query-group cohort fixes the existing development seed')
    base = base_builder(spec, 'native', assets, source, output, seed)
    rows = []
    for lane, task in enumerate(('w2_rotating_wooden_bird', 'w2_tracking_delivery_cart')):
        template = next(x for x in base if x['scenario'] == task)
        for method in ('native', 'old_block64_fast', 'shared', 'split_shared', 'split_specific'):
            name = f'{task}__s{seed}__qgroup_{method}'
            cmd = list(template['cmd'])
            def put(key, value):
                if key in cmd: cmd[cmd.index(key)+1] = str(value)
                else: cmd.extend([key, str(value)])
            put('--output', output/name)
            if method != 'native':
                put('--wave2-method', 'w2_steady_sparse')
                put('--wave2-selector', 'query_sum_batch4')
                put('--wave2-preparation', 'geometry_cache')
            if method in ('shared','split_shared','split_specific'):
                put('--wave2-query-groups', method)
            rows.append(dict(id=name,scenario=task,method=method,cmd=cmd,latent_frames=128,
                cohort_pair=lane,repeat_reason='matched_whole_frame_and_query_execution_controls',formal_holdout=False))
    return rows
