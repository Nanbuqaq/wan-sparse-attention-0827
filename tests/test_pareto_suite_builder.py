from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.build_pareto_suites import (
    BASIC_CATEGORIES,
    DENSITIES,
    LONG_CATEGORIES,
    REFRESH_POLICIES,
    ROPE_POLICIES,
)


def test_frozen_pareto_axes_match_protocol():
    assert DENSITIES == (0.05, 0.10, 0.15, 0.20, 0.25)
    assert REFRESH_POLICIES == ("per_chunk", "per_step")
    assert ROPE_POLICIES == (
        "upstream_zero",
        "recency_rank",
        "clipped_relative_age",
    )
    assert BASIC_CATEGORIES == ("identity_scene", "irreversible_state")
    assert LONG_CATEGORIES == ("irreversible_state", "fast_motion")


def test_one_selected_method_builds_28_sparse_and_10_dense_cases(tmp_path):
    categories = (
        "identity_scene",
        "irreversible_state",
        "human_action",
        "fast_motion",
    )
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
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps({"selected_methods": ["block64_history"]}), encoding="utf-8"
    )
    calibration = tmp_path / "params.json"
    calibration.write_text(
        json.dumps(
            {
                "status": "frozen_before_method_smoke",
                "method_params": {},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/build_pareto_suites.py"),
            "--frozen-prompts",
            str(frozen),
            "--selection",
            str(selection),
            "--calibration",
            str(calibration),
            "--commit",
            "a" * 40,
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    suite = json.loads((output / "rag_pareto_expansion.json").read_text())
    dense = json.loads((output / "rag_dense_pareto_expansion.json").read_text())
    expected = json.loads((output / "expected_pareto_expansion.json").read_text())
    assert len(suite["cases"]) == 28
    assert len(dense["cases"]) == 10
    assert len(expected["cases"]) == 38
