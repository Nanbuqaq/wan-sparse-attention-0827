from __future__ import annotations

from adapters.longlive_sparse.profiling import classify_bottleneck, process_rss_bytes
from adapters.longlive_sparse.profiling import profiled
import pytest


def test_nvtx_is_opt_in_and_balanced(monkeypatch):
    events = []
    monkeypatch.delenv('LONGLIVE_NVTX', raising=False)
    monkeypatch.setattr('torch.cuda.is_available', lambda: events.append('cuda') or True)
    monkeypatch.setattr('torch.cuda.nvtx.range_push', lambda name: events.append(name))
    monkeypatch.setattr('torch.cuda.nvtx.range_pop', lambda: events.append('pop'))

    @profiled('scope')
    def fail():
        raise ValueError('expected')

    with pytest.raises(ValueError):
        fail()
    assert events == []
    monkeypatch.setenv('LONGLIVE_NVTX', '1')
    with pytest.raises(ValueError):
        fail()
    assert events == ['cuda', 'scope', 'pop']


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
