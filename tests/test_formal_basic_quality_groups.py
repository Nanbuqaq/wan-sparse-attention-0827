from __future__ import annotations

import scripts.evaluate_formal_basic_quality as quality
from scripts.evaluate_formal_basic_quality import (
    assign_groups,
    build_quality_groups,
    physical_gpu_locks,
)


def case(method, *, status="pass", video="/bin/sh", case_id=None):
    return {
        "id": case_id or method,
        "method": method,
        "status": status,
        "video": video,
        "commit": "a" * 40,
        "prompt_id": "p",
        "seed": 1,
        "latent_frames": 120,
    }


def test_quality_groups_keep_native_and_rag_pairing_separate():
    groups, skipped = build_quality_groups(
        [
            case("native_dense"),
            case("native_block"),
            case("rag_dense"),
            case("block64_history"),
            case("coverage_cluster_history", status="negative"),
            case("token_oracle", status="fail", video="missing"),
        ]
    )
    by_baseline = {group["baseline"]["method"]: group for group in groups}
    assert {item["method"] for item in by_baseline["native_dense"]["cases"]} == {
        "native_dense",
        "native_block",
    }
    assert {item["method"] for item in by_baseline["rag_dense"]["cases"]} == {
        "rag_dense",
        "block64_history",
        "coverage_cluster_history",
    }
    assert skipped == ["token_oracle"]


def test_greedy_assignment_balances_large_quality_groups():
    groups = [
        {"group_id": "large-a", "cases": [1] * 20},
        {"group_id": "large-b", "cases": [1] * 20},
        {"group_id": "small-a", "cases": [1] * 2},
        {"group_id": "small-b", "cases": [1] * 2},
    ]
    lanes = assign_groups(groups, ["0", "1"])
    assert sorted(sum(len(group["cases"]) for group in lane) for lane in lanes) == [22, 22]


def test_quality_group_allows_multiple_frozen_configs_of_one_method():
    groups, skipped = build_quality_groups(
        [
            case("rag_dense"),
            case("scope_ar", case_id="scope-density-10"),
            case("scope_ar", case_id="scope-density-25"),
        ]
    )
    assert skipped == []
    assert len(groups) == 1
    assert [item["id"] for item in groups[0]["cases"]] == [
        "rag_dense",
        "scope-density-10",
        "scope-density-25",
    ]


def test_dual_gpu_quality_uses_only_physical_locks(monkeypatch, tmp_path):
    monkeypatch.setattr(
        quality,
        "gpu_rows",
        lambda: [
            {"index": 0, "memory": 0, "utilization": 0},
            {"index": 1, "memory": 0, "utilization": 0},
        ],
    )
    with physical_gpu_locks(
        ["0", "1"],
        max_memory_mib=1024,
        max_utilization=20,
        lock_root=tmp_path,
    ) as locks:
        assert locks == [
            str(tmp_path / "wan_sparse_gpu_0.lock"),
            str(tmp_path / "wan_sparse_gpu_1.lock"),
        ]
