from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_formal_suite_has_38_commit_aware_cases_and_public_safe_provenance(tmp_path):
    categories = (
        "identity_scene",
        "irreversible_state",
        "human_action",
        "fast_motion",
    )
    frozen = tmp_path / "frozen_prompts.json"
    frozen.write_text(
        json.dumps(
            {
                "artifact_id": "frozen_prompts",
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
    calibration = tmp_path / "method_params.json"
    calibration.write_text(
        json.dumps(
            {
                "status": "frozen_before_method_smoke",
                "method_params": {},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "formal"
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/build_formal_suites.py"),
            "--frozen-prompts",
            str(frozen),
            "--calibration",
            str(calibration),
            "--commit",
            commit,
            "--output-dir",
            str(output),
        ],
        cwd=root,
        check=True,
    )
    expected = json.loads((output / "expected_basic_477.json").read_text())
    assert len(expected["cases"]) == 38
    assert len({case["case_key_sha256"] for case in expected["cases"]}) == 38
    suite_text = (output / "rag_basic_477.json").read_text()
    assert "/home/" not in suite_text
    internal_mount = "/" + "kaimm" + "-distill"
    assert internal_mount not in suite_text
