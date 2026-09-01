from __future__ import annotations

import torch

from scripts.audit_qkv_captures import audit_capture_grid, parse_ints


def write_capture(path, *, layer, start, finite=True):
    query = torch.ones((1, 6, 2, 4), dtype=torch.bfloat16)
    key = torch.ones((1, 9, 2, 4), dtype=torch.bfloat16)
    value = torch.ones((1, 9, 2, 4), dtype=torch.bfloat16)
    if not finite:
        query[0, 0, 0, 0] = float("nan")
    torch.save(
        {
            "layer": layer,
            "current_start": start,
            "query": query,
            "key": key,
            "value": value,
        },
        path,
    )


def test_parse_ints_requires_unique_nonempty_values():
    assert parse_ints("0,9,19,29") == [0, 9, 19, 29]


def test_complete_capture_grid_passes(tmp_path):
    for layer in (0, 1):
        for start in (3, 7):
            write_capture(
                tmp_path / f"layer{layer:02d}_start{start:08d}.pt",
                layer=layer,
                start=start,
            )
    result = audit_capture_grid(
        tmp_path,
        layers=[0, 1],
        starts=[3, 7],
        query_tokens=6,
        heads=2,
        head_dim=4,
        frame_tokens=3,
    )
    assert result["status"] == "pass"
    assert len(result["records"]) == 4
    assert {record["history_frames"] for record in result["records"]} == {3}


def test_missing_and_nonfinite_captures_fail(tmp_path):
    write_capture(
        tmp_path / "layer00_start00000003.pt", layer=0, start=3, finite=False
    )
    result = audit_capture_grid(
        tmp_path,
        layers=[0],
        starts=[3, 7],
        query_tokens=6,
        heads=2,
        head_dim=4,
        frame_tokens=3,
    )
    assert result["status"] == "fail"
    assert any("missing captures" in error for error in result["errors"])
    assert any("non-finite" in error for error in result["errors"])
