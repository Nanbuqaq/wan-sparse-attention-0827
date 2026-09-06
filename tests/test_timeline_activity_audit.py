from scripts.audit_generator_timeline import duration, intersection


def test_nested_cpu_and_overlapping_gpu_spans_are_not_added_twice():
    assert duration([(0, 10), (2, 5), (6, 12)]) == 12
    assert intersection([(0, 10), (2, 5)], [(3, 7), (6, 8)]) == 5
    assert intersection([(0, 2), (8, 10)], [(3, 7)]) == 0
