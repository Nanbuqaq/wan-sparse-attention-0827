import pytest

from scripts.audit_full_flow_timeline import attribute_runtime, summarize


def test_async_kernel_is_attributed_to_launch_not_execution_timestamp():
    scopes = [(0, 1000, "generation.complete", 7), (10, 20, "transformer.ffn", 7)]
    assignments = attribute_runtime(scopes, [(5, 15, 7), (6, 30, 7)])
    # Correlation 5 may execute at 50..70, outside the CPU FFN range.
    assert assignments[5] == ("transformer.ffn", "generation.complete")
    assert assignments[6] == ("generation.complete", "generation.complete")


def test_activity_union_does_not_double_count_overlap_and_includes_memset():
    result = summarize([(50, 70, "kernel", 0), (60, 80, "kernel", 0),
                        (55, 65, "H2D", 100), (90, 100, "memset", 64)])
    assert result["GPU_kernel_service_sum_s"] == pytest.approx(40/1e9)
    assert result["GPU_kernel_active_union_s"] == pytest.approx(30/1e9)
    assert result["GPU_all_activity_union_s"] == pytest.approx(40/1e9)
    assert result["H2D"]["overlap_with_kernels_s"] == pytest.approx(10/1e9)


def test_scopes_on_another_thread_do_not_claim_the_kernel():
    assignments = attribute_runtime([(0, 1000, "generation.complete", 7)], [(1, 20, 9)])
    assert assignments[1] == ("unscoped", "unscoped")
