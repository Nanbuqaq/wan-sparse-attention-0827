from scripts.build_mentor_kernel_catalog import resource_upper_bound


def test_static_limits_are_explicitly_not_measured_occupancy():
    props=dict(max_threads_per_multi_processor=1536,regs_per_multiprocessor=65536,
               shared_memory_per_multiprocessor=102400,warp_size=32)
    row=resource_upper_bound(128,64,65536,props)
    assert row['CTA_upper_bound']==1
    assert row['warp_occupancy_upper_bound']==4/48
    assert row['limiting_resources']==['shared_limit']
    assert not row['measured_occupancy']
