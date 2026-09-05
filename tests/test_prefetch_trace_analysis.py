from __future__ import annotations

from pathlib import Path

import torch

from adapters.longlive_sparse.route_plan import HistoryRoutePlan
from scripts.analyze_prefetch_trace import analyze


def _plan(tokens: list[int]) -> HistoryRoutePlan:
    width = len(tokens)
    return HistoryRoutePlan(
        method="test",
        routing_stage="pre-transfer",
        query_labels=torch.tensor([[[0, 0]]]),
        query_group_sizes=torch.tensor([[[2]]]),
        union_frame_ids=torch.tensor([[[1] * width]]),
        union_token_ids=torch.tensor([[tokens]]),
        group_union_indices=torch.arange(width).view(1, 1, 1, width),
        group_history_counts=torch.tensor([[[width]]]),
        candidate_history_tokens=256,
        query_tokens=2,
        exact_k_tokens=2,
        target_history_density=0.25,
    )


def _write(path: Path, *, layer: int, tokens: list[int]) -> None:
    plan = _plan(tokens)
    torch.save(
        {
            "layer": layer,
            "current_start": 100,
            "denoising_pass": 0,
            "route_plan": plan.state_dict(),
        },
        path,
    )


def test_prefetch_trace_reports_bounded_completion(tmp_path: Path) -> None:
    first = tmp_path / "layer00_start00000100_pass00.pt"
    second = tmp_path / "layer01_start00000100_pass00.pt"
    _write(first, layer=0, tokens=[0, 64])
    _write(second, layer=1, tokens=[64, 128])
    payload = analyze([first, second], bytes_per_block=100)
    assert payload["records"] == 1
    assert payload["mean_prediction_recall"] == 0.5
    assert payload["total_extra_bytes"] == 100
    assert payload["total_miss_bytes"] == 100
    assert payload["all_final_execution_exact_actual"] is True
