from scripts.benchmark_offload_overlap import overlap_ms


def test_overlap_deduplicates_nested_service_intervals():
    assert overlap_ms([(0,4),(1,3)],[(2,6)])==2
    assert overlap_ms([(0,1),(2,3)],[(1,2)])==0
    assert overlap_ms([(0,2),(3,5)],[(1,4)])==2
