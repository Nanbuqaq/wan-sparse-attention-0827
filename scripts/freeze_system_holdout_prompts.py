#!/usr/bin/env python3
"""Freeze the new four-prompt system holdout before any sparse video."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters.longlive_sparse.formal_gate import validate_system_holdouts


SCORE_FIELDS = [
    "category_completion_0to2",
    "subject_consistency_0to2",
    "background_consistency_0to2",
    "continuous_motion_0to2",
    "freeze_flicker_cut_0to2",
]
COUNT_FIELDS = ["state_reset_count", "freeze_count", "camera_cut_count"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "pass"}:
        return True
    if normalized in {"false", "0", "no", "fail"}:
        return False
    raise ValueError(f"invalid boolean review value: {value!r}")


def freeze(
    review_path: Path,
    candidate_path: Path,
    fixed_prompt_path: Path,
    dense_audit_path: Path,
) -> dict:
    review = pd.read_csv(review_path)
    forbidden = [column for column in review.columns if "sparse" in column.lower()]
    if forbidden:
        raise ValueError(f"state prompt freeze cannot consume sparse columns: {forbidden}")
    required = {
        "case_id",
        "commit",
        "prompt_id",
        "category",
        "seed",
        "runtime",
        "technical_pass",
        "decoded_frames",
        "prompt",
        *SCORE_FIELDS,
        *COUNT_FIELDS,
    }
    missing = sorted(required - set(review.columns))
    if missing:
        raise ValueError(f"missing state review columns: {missing}")
    manifest = json.loads(candidate_path.read_text(encoding="utf-8"))
    expected_seeds = {int(value) for value in manifest["seeds"]}
    review["technical_pass"] = review["technical_pass"].map(_parse_bool)
    for field in SCORE_FIELDS:
        review[field] = pd.to_numeric(review[field], errors="raise")
        if (~review[field].isin([0, 1, 2])).any():
            raise ValueError(f"{field} must contain only 0, 1 or 2")
    for field in COUNT_FIELDS:
        review[field] = pd.to_numeric(review[field], errors="raise")
        if (review[field] < 0).any():
            raise ValueError(f"{field} must be non-negative")
    if set(review["runtime"]) != {"rag_dense"}:
        raise ValueError("state prompt freeze accepts RAG Dense only")
    commits = {str(value) for value in review["commit"]}
    if len(commits) != 1 or len(next(iter(commits))) != 40:
        raise ValueError("state review requires one shared full commit")
    if set(int(value) for value in review["seed"]) != expected_seeds:
        raise ValueError("state review seed set does not match candidate manifest")
    review["video_total"] = review[SCORE_FIELDS].sum(axis=1)
    rankings = []
    for (prompt_id, prompt), rows in review.groupby(["prompt_id", "prompt"], sort=False):
        seeds = {int(value) for value in rows["seed"]}
        technical = bool(rows["technical_pass"].all())
        decoded = bool((rows["decoded_frames"] == manifest["expected_pixel_frames"]).all())
        category_each = bool((rows["category_completion_0to2"] >= 1).all())
        category_any = bool((rows["category_completion_0to2"] >= 2).any())
        no_events = bool((rows[COUNT_FIELDS].sum(axis=1) == 0).all())
        worst = float(rows["video_total"].min())
        eligible = (
            seeds == expected_seeds
            and technical
            and decoded
            and category_each
            and category_any
            and no_events
            and worst >= float(manifest["gate"]["worst_seed_total_min"])
        )
        rankings.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "seeds": len(seeds),
                "technical_pass": technical,
                "decoded_complete": decoded,
                "category_completion_each_seed": category_each,
                "category_completion_any_seed": category_any,
                "no_reset_freeze_cut": no_events,
                "two_seed_average": float(rows["video_total"].mean()),
                "worst_seed": worst,
                "eligible": eligible,
            }
        )
    rankings.sort(
        key=lambda item: (
            not item["eligible"],
            -item["two_seed_average"],
            -item["worst_seed"],
            item["prompt_id"],
        )
    )
    eligible = [item for item in rankings if item["eligible"]]
    fixed_manifest = json.loads(fixed_prompt_path.read_text(encoding="utf-8"))
    by_id = {item["prompt_id"]: item for item in fixed_manifest["candidates"]}
    fixed_ids = manifest["fixed_non_state_holdouts"]
    fixed_prompts = [by_id[prompt_id] for prompt_id in fixed_ids]
    if eligible:
        selected = {
            "prompt_id": eligible[0]["prompt_id"],
            "category": "irreversible_state",
            "prompt": eligible[0]["prompt"],
            "holdout_role": "new_dense_screened_holdout",
        }
    else:
        fallback_id = manifest["gate"]["fallback_prompt_id"]
        selected = {
            **by_id[fallback_id],
            "holdout_role": manifest["gate"]["fallback_role"],
        }
    prompts = [
        {**item, "holdout_role": "old_dense_screened_never_sparse_evaluated"}
        for item in fixed_prompts
    ] + [selected]
    payload = {
        "artifact_id": "longlive_system_holdout_prompts_v1",
        "status": "frozen_before_sparse_system_video",
        "dense_screen_commit": next(iter(commits)),
        "selection_source": {
            "artifact_id": review_path.name,
            "sha256": _sha256(review_path),
        },
        "dense_terminal_audit": {
            "artifact_id": dense_audit_path.name,
            "sha256": _sha256(dense_audit_path),
        },
        "candidate_manifest": {
            "artifact_id": candidate_path.name,
            "sha256": _sha256(candidate_path),
        },
        "sparse_results_used": False,
        "formal_477_seeds": [20260908, 20260909],
        "formal_957_seeds": [20260910],
        "ranking_rule": manifest["gate"]["ranking"],
        "state_rankings": rankings,
        "selected_state_prompt_id": selected["prompt_id"],
        "state_holdout_role": selected["holdout_role"],
        "stress_only_prompt_id": manifest["stress_only_prompt_id"],
        "prompts": prompts,
    }
    errors = validate_system_holdouts(payload)
    if errors:
        raise RuntimeError("generated invalid system holdouts: " + "; ".join(errors))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True)
    parser.add_argument(
        "--candidates", default="configs/system/state_prompt_candidates.json"
    )
    parser.add_argument(
        "--fixed-prompts", default="configs/prompts/dense_candidates.json"
    )
    parser.add_argument("--dense-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = freeze(
        Path(args.review),
        Path(args.candidates),
        Path(args.fixed_prompts),
        Path(args.dense_audit),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
