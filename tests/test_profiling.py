from __future__ import annotations

from adapters.longlive_sparse.profiling import classify_bottleneck, process_rss_bytes


def test_process_rss_is_available_or_explicitly_none() -> None:
    value = process_rss_bytes()
    assert value is None or value > 0


def test_bottleneck_classifier_allows_mixed_and_incomplete() -> None:
    result = classify_bottleneck(
        {
            "cpu_route_gather": 3.0,
            "host_device_transfer": 2.7,
            "attention": 2.6,
            "pipeline_bubble": 1.6,
        },
        total_critical_path_s=10.0,
    )
    assert "cpu-bound" in result["labels"]
    assert "transfer-bound" in result["labels"]
    assert "attention-bound-incomplete-counters" in result["labels"]
    assert "pipeline-bubble" in result["labels"]


def test_hardware_counters_distinguish_hbm_and_compute() -> None:
    hbm = classify_bottleneck(
        {"attention": 5.0},
        total_critical_path_s=10.0,
        dram_throughput_fraction=0.8,
        sm_throughput_fraction=0.5,
    )
    compute = classify_bottleneck(
        {"attention": 5.0},
        total_critical_path_s=10.0,
        dram_throughput_fraction=0.6,
        sm_throughput_fraction=0.8,
    )
    assert hbm["labels"] == ["hbm-bound"]
    assert compute["labels"] == ["compute-bound"]
