from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


FIELDS = [
    "category_completion_0to2",
    "subject_consistency_0to2",
    "background_consistency_0to2",
    "continuous_motion_0to2",
    "freeze_flicker_cut_0to2",
]


def test_dense_prompt_freeze_uses_pair_minimum_then_average_and_worst_seed(tmp_path):
    categories = ["identity_scene", "irreversible_state", "human_action", "fast_motion"]
    rows = []
    for category in categories:
        for suffix in ("a", "b"):
            prompt_id = f"{category}_{suffix}"
            for seed in (20260826, 20260827):
                for runtime in ("native_dense", "rag_dense"):
                    score = 2
                    if suffix == "a" and runtime == "rag_dense" and seed == 20260827:
                        score = 1
                    if suffix == "b":
                        score = 1
                    rows.append(
                        {
                            "case_id": f"{runtime}-{prompt_id}-{seed}",
                            "commit": "a" * 40,
                            "prompt_id": prompt_id,
                            "category": category,
                            "seed": seed,
                            "runtime": runtime,
                            "technical_pass": True,
                            "prompt": f"prompt {prompt_id}",
                            **{field: score for field in FIELDS},
                        }
                    )
    review = tmp_path / "review.csv"
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "frozen.json"
    script = Path(__file__).resolve().parents[1] / "scripts/freeze_dense_prompts.py"
    subprocess.run(
        [sys.executable, str(script), "--review", str(review), "--output", str(output)],
        check=True,
    )
    frozen = json.loads(output.read_text(encoding="utf-8"))
    assert {item["prompt_id"] for item in frozen["prompts"]} == {
        f"{category}_a" for category in categories
    }
    assert frozen["sparse_results_used"] is False
    assert frozen["selection_source"]["artifact_id"] == "review.csv"
