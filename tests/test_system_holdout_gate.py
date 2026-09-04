from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from adapters.longlive_sparse.formal_gate import (
    load_frozen_system_holdouts,
    validate_system_holdouts,
)
from scripts.freeze_system_holdout_prompts import freeze
from scripts.run_loaded_dense_screen import sharded_candidates


SCORES = [
    "category_completion_0to2",
    "subject_consistency_0to2",
    "background_consistency_0to2",
    "continuous_motion_0to2",
    "freeze_flicker_cut_0to2",
]


def _candidate_manifest(path: Path) -> dict:
    payload = {
        "expected_pixel_frames": 477,
        "seeds": [20260905, 20260906],
        "candidates": [
            {"prompt_id": "state_a", "category": "irreversible_state", "prompt": "A"},
            {"prompt_id": "state_b", "category": "irreversible_state", "prompt": "B"},
        ],
        "gate": {
            "worst_seed_total_min": 8,
            "ranking": ["two_seed_average desc", "worst_seed desc", "prompt_id asc"],
            "fallback_prompt_id": "state_water_pour",
            "fallback_role": "known_regression_not_independent_holdout",
        },
        "fixed_non_state_holdouts": [
            "identity_mars_astronaut",
            "human_glassblower",
            "fast_fox_snow",
        ],
        "stress_only_prompt_id": "state_melting_candle",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _fixed_prompts(path: Path) -> None:
    candidates = []
    for prompt_id, category in (
        ("identity_mars_astronaut", "identity_scene"),
        ("human_glassblower", "human_action"),
        ("fast_fox_snow", "fast_motion"),
        ("state_water_pour", "irreversible_state"),
    ):
        candidates.append(
            {"prompt_id": prompt_id, "category": category, "prompt": prompt_id}
        )
    path.write_text(json.dumps({"candidates": candidates}), encoding="utf-8")


def _review(path: Path, *, eligible: bool) -> None:
    rows = []
    for prompt_id in ("state_a", "state_b"):
        for seed in (20260905, 20260906):
            category = 2 if prompt_id == "state_a" and eligible else 0
            row = {
                "case_id": f"{prompt_id}-{seed}",
                "commit": "a" * 40,
                "prompt_id": prompt_id,
                "category": "irreversible_state",
                "seed": seed,
                "runtime": "rag_dense",
                "technical_pass": True,
                "decoded_frames": 477,
                "video": f"{prompt_id}.mp4",
                "prompt": "A" if prompt_id == "state_a" else "B",
                "review_notes": "",
                "state_reset_count": 0,
                "freeze_count": 0,
                "camera_cut_count": 0,
                **{field: 2 for field in SCORES},
            }
            row["category_completion_0to2"] = category
            rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_case_axis_sharding_assigns_one_dense_case_per_lane() -> None:
    manifest = {
        "seeds": [1, 2],
        "candidates": [
            {"prompt_id": "a", "prompt": "A"},
            {"prompt_id": "b", "prompt": "B"},
        ],
    }
    shards = [
        sharded_candidates(
            manifest, shard_axis="case", shard_index=index, shard_count=4
        )
        for index in range(4)
    ]
    assert all(len(shard) == 1 for shard in shards)
    assert {
        (shard[0]["prompt_id"], shard[0]["_seeds"][0]) for shard in shards
    } == {("a", 1), ("a", 2), ("b", 1), ("b", 2)}


def test_freeze_selects_new_state_prompt_and_passes_gate(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.json"
    fixed = tmp_path / "fixed.json"
    review = tmp_path / "review.csv"
    audit = tmp_path / "audit.json"
    _candidate_manifest(candidates)
    _fixed_prompts(fixed)
    _review(review, eligible=True)
    audit.write_text("{}", encoding="utf-8")
    payload = freeze(review, candidates, fixed, audit)
    assert payload["selected_state_prompt_id"] == "state_a"
    assert payload["state_holdout_role"] == "new_dense_screened_holdout"
    assert validate_system_holdouts(payload) == []
    output = tmp_path / "system_holdouts.json"
    output.write_text(json.dumps(payload), encoding="utf-8")
    assert load_frozen_system_holdouts(output)["artifact_id"] == payload["artifact_id"]


def test_freeze_falls_back_to_water_pour_without_pretending_holdout(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.json"
    fixed = tmp_path / "fixed.json"
    review = tmp_path / "review.csv"
    audit = tmp_path / "audit.json"
    _candidate_manifest(candidates)
    _fixed_prompts(fixed)
    _review(review, eligible=False)
    audit.write_text("{}", encoding="utf-8")
    payload = freeze(review, candidates, fixed, audit)
    assert payload["selected_state_prompt_id"] == "state_water_pour"
    assert payload["state_holdout_role"] == "known_regression_not_independent_holdout"


def test_sparse_gate_fails_closed_when_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="blocked"):
        load_frozen_system_holdouts(tmp_path / "missing.json")


def test_state_screen_batch_shell_parses() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["bash", "-n", str(root / "scripts/inferhub_batch_system_state_screen.sh")],
        check=True,
    )
