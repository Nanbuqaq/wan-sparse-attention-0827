from __future__ import annotations

from scripts.evaluate_formal_basic_quality import assign_groups, build_quality_groups


def case(method, *, status="pass", video="/bin/sh"):
    return {
        "id": method,
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
