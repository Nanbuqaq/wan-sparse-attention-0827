from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


METHODS = [
    "coverage_cluster_history",
    "vaware_cluster_history",
    "transfer_vaware_hybrid_history",
]


def test_proposed_params_freeze_after_eight_case_isolated_gate(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "status": "frozen_before_method_smoke",
                "method_params": {"svg2_ar": {"q_clusters": 300}},
            }
        ),
        encoding="utf-8",
    )
    qkv = tmp_path / "qkv.json"
    qkv.write_text(
        json.dumps(
            {
                "status": "qkv_calibrated_long_video_freeze_pending",
                "formal_prompts_used": False,
                "analysis_worktree_clean": True,
                "online_information_boundary": ["Q summaries", "K/V prototypes"],
                "qkv_selected_candidates": {
                    method: {
                        "candidate_id": f"{method}_candidate",
                        "method_params": {"remote_clusters": 128},
                    }
                    for method in METHODS
                },
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "pass",
                "expected_cases": 8,
                "terminal_cases": 8,
                "errors": [],
            }
        ),
        encoding="utf-8",
    )
    metrics = tmp_path / "metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "case_id", "status"])
        writer.writeheader()
        for method in METHODS:
            for case in range(2):
                writer.writerow(
                    {"method": method, "case_id": f"{method}_{case}", "status": "pass"}
                )
    output = tmp_path / "frozen.json"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/freeze_proposed_method_params.py"),
            "--base-params",
            str(base),
            "--qkv-calibration",
            str(qkv),
            "--long-audit",
            str(audit),
            "--case-metrics",
            str(metrics),
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "frozen_before_formal_long_video"
    assert set(METHODS).issubset(payload["method_params"])
    assert payload["method_params"]["svg2_ar"] == {"q_clusters": 300}
    assert payload["proposed_method_freeze"]["output_residual_role"] == "offline_teacher_only"


def test_long_calibration_builder_emits_two_rag_pairs_per_method(tmp_path):
    qkv = tmp_path / "qkv.json"
    qkv.write_text(
        json.dumps(
            {
                "status": "qkv_calibrated_long_video_freeze_pending",
                "formal_prompts_used": False,
                "analysis_worktree_clean": True,
                "qkv_selected_candidates": {
                    method: {
                        "candidate_id": f"{method}_candidate",
                        "method_params": {"remote_clusters": 128},
                    }
                    for method in METHODS
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "suite"
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/build_proposed_long_calibration_suite.py"),
            "--qkv-calibration",
            str(qkv),
            "--output-dir",
            str(output),
            "--commit",
            commit,
        ],
        cwd=root,
        check=True,
    )
    sparse = json.loads((output / "proposed_long_sparse.json").read_text())
    expected = json.loads((output / "expected_proposed_long.json").read_text())
    assert sparse["formal_prompts_used"] is False
    assert sparse["methods"] == METHODS
    assert len(sparse["cases"]) == 2
    assert len(expected["cases"]) == 8


def test_final_long_confirmation_has_native_and_rag_pairing_panel(tmp_path):
    categories = [
        "identity_scene",
        "irreversible_state",
        "human_action",
        "fast_motion",
    ]
    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps(
            {
                "status": "frozen",
                "sparse_results_used": False,
                "prompts": [
                    {
                        "prompt_id": category,
                        "category": category,
                        "prompt": f"prompt {category}",
                    }
                    for category in categories
                ],
            }
        ),
        encoding="utf-8",
    )
    params = tmp_path / "params.json"
    params.write_text(
        json.dumps(
            {
                "status": "frozen_before_formal_long_video",
                "method_params": {
                    "transfer_vaware_hybrid_history": {"remote_clusters": 128}
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "final"
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/build_final_long_confirmation_suite.py"),
            "--frozen-prompts",
            str(frozen),
            "--method-params",
            str(params),
            "--output-dir",
            str(output),
            "--commit",
            commit,
        ],
        cwd=root,
        check=True,
    )
    expected = json.loads((output / "expected_final_long.json").read_text())
    assert len(expected["cases"]) == 20
    methods = {case["method"] for case in expected["cases"]}
    assert methods == {
        "native_dense",
        "native_block",
        "rag_dense",
        "block64_history",
        "transfer_vaware_hybrid_history",
    }
