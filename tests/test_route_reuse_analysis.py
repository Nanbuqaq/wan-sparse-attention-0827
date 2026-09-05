from __future__ import annotations

import json
from pathlib import Path

import torch

from adapters.longlive_sparse.route_plan import HistoryRoutePlan
from scripts.analyze_route_reuse import analyze


def _plan(token: int) -> HistoryRoutePlan:
    return HistoryRoutePlan(
        method="block64_history",
        routing_stage="pre-transfer",
        query_labels=torch.tensor([[[0, 0]]]),
        query_group_sizes=torch.tensor([[[2]]]),
        union_frame_ids=torch.tensor([[[1]]]),
        union_token_ids=torch.tensor([[[token]]]),
        group_union_indices=torch.tensor([[[[0]]]]),
        group_history_counts=torch.tensor([[[1]]]),
        candidate_history_tokens=4,
        query_tokens=2,
        exact_k_tokens=2,
        target_history_density=0.25,
    )


def _write(path: Path, *, layer: int, denoising_pass: int, token: int, hit: bool) -> None:
    plan = _plan(token)
    torch.save(
        {
            "layer": layer,
            "current_start": 100,
            "denoising_pass": denoising_pass,
            "route_plan": plan.state_dict(),
            "route_plan_sha256": plan.digest(),
            "transfer_plan_sha256": "a" * 64,
            "cache_hit": hit,
            "archive_epoch": 1,
            "storage_version": 2,
        },
        path,
    )


def test_reuse_analysis_separates_denoising_and_layer_axes(tmp_path: Path) -> None:
    paths = []
    for layer in (0, 1):
        for denoising_pass in (0, 1):
            path = tmp_path / f"layer{layer}_pass{denoising_pass}.pt"
            _write(
                path,
                layer=layer,
                denoising_pass=denoising_pass,
                token=(0 if layer == 0 else denoising_pass),
                hit=denoising_pass > 0,
            )
            paths.append(path)
    payload = analyze(paths)
    assert payload["records"] == 4
    layer0 = next(item for item in payload["denoising_axis"] if item["layer"] == 0)
    layer1 = next(item for item in payload["denoising_axis"] if item["layer"] == 1)
    assert layer0["same_route_sha_all"] is True
    assert layer0["cache_hits"] == 1
    assert layer1["min_jaccard_vs_first"] == 0.0
    assert len(payload["layer_axis"]) == 2
