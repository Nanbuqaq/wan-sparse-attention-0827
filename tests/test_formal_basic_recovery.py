from __future__ import annotations

import json
from pathlib import Path

from scripts.recover_formal_basic_results import (
    artifact_bucket,
    result_roots,
    rewrite_local_states,
    validate_terminal_source,
)


def test_artifacts_are_partitioned_without_losing_relative_paths(tmp_path):
    assert artifact_bucket(tmp_path / "case" / "video.mp4") == "videos"
    assert artifact_bucket(tmp_path / "case" / "latents.pt") == "latents"
    assert artifact_bucket(tmp_path / "qkv_captures" / "layer00.pt") == "captures"
    assert artifact_bucket(tmp_path / "lane0.log") == "logs"
    assert artifact_bucket(tmp_path / "review.png") == "reviews"
    assert artifact_bucket(tmp_path / "case_state.json") == "manifests"


def test_terminal_source_accepts_explicit_fail_evidence(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    cases = [
        {"id": "a", "status": "fail", "failure_reason": "negative control"},
        {"id": "b", "status": "fail", "failure_reason": "negative control"},
    ]
    (control / "expected_basic_477.json").write_text(
        json.dumps({"cases": [{"id": "a"}, {"id": "b"}]}), encoding="utf-8"
    )
    (tmp_path / "merged_case_states.json").write_text(
        json.dumps({"cases": cases}), encoding="utf-8"
    )
    (tmp_path / "terminal_state_audit.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "expected_cases": 2,
                "terminal_cases": 2,
                "pass_cases": 0,
                "negative_cases": 0,
                "fail_cases": 2,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "batch_status.json").write_text(
        json.dumps({"terminal_audit_completed": True, "lane_statuses": [1]}),
        encoding="utf-8",
    )
    (tmp_path / "SHA256SUMS.txt").write_text("", encoding="utf-8")
    summary = validate_terminal_source(tmp_path, expected_cases=2)
    assert summary["statuses"] == {"fail": 2}


def test_terminal_source_accepts_an_alternate_expected_manifest(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    (control / "expected_final_long.json").write_text(
        json.dumps({"cases": [{"id": "a"}]}), encoding="utf-8"
    )
    (tmp_path / "merged_case_states.json").write_text(
        json.dumps(
            {"cases": [{"id": "a", "status": "fail", "failure_reason": "x"}]}
        ),
        encoding="utf-8",
    )
    (tmp_path / "terminal_state_audit.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "expected_cases": 1,
                "terminal_cases": 1,
                "fail_cases": 1,
                "pass_cases": 0,
                "negative_cases": 0,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "batch_status.json").write_text(
        json.dumps({"terminal_audit_completed": True, "lane_statuses": [1]}),
        encoding="utf-8",
    )
    (tmp_path / "SHA256SUMS.txt").write_text("", encoding="utf-8")
    summary = validate_terminal_source(
        tmp_path,
        expected_cases=1,
        expected_relative=Path("control/expected_final_long.json"),
    )
    assert summary["statuses"] == {"fail": 1}


def test_local_state_paths_follow_partitioned_results_roots(tmp_path):
    source = tmp_path / "source"
    source_case = source / "lane0" / "method" / "case"
    roots = result_roots(tmp_path / "results", run_id="run", date_tag="20260901")
    payload = {
        "cases": [
            {
                "id": "case",
                "status": "pass",
                "video": str(source_case / "video.mp4"),
                "stats": str(source_case / "sparse_history_stats.json"),
                "config": str(source_case / "case_config.json"),
            }
        ]
    }
    local = rewrite_local_states(payload, source_root=source, roots=roots)["cases"][0]
    assert local["video"].startswith(str(roots["videos"]))
    assert local["latent"].startswith(str(roots["latents"]))
    assert local["stats"].startswith(str(roots["manifests"]))
    assert local["config"].startswith(str(roots["manifests"]))
