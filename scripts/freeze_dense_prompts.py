#!/usr/bin/env python3
"""Freeze formal prompts using Dense-only two-seed review evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


CATEGORIES = ["identity_scene", "irreversible_state", "human_action", "fast_motion"]
EXPECTED_SEEDS = {20260826, 20260827}
EXPECTED_RUNTIMES = {"native_dense", "rag_dense"}
SCORE_FIELDS = [
    "category_completion_0to2",
    "subject_consistency_0to2",
    "background_consistency_0to2",
    "continuous_motion_0to2",
    "freeze_flicker_cut_0to2",
]


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "pass"}:
        return True
    if normalized in {"false", "0", "no", "fail"}:
        return False
    raise ValueError(f"invalid boolean review value: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-output")
    args = parser.parse_args()
    review = pd.read_csv(args.review)
    forbidden = [column for column in review.columns if "sparse" in column.lower()]
    if forbidden:
        raise ValueError(f"Dense prompt freeze cannot consume sparse columns: {forbidden}")
    required = {
        "case_id",
        "commit",
        "prompt_id",
        "category",
        "seed",
        "runtime",
        "technical_pass",
        "prompt",
        *SCORE_FIELDS,
    }
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"missing review columns: {sorted(missing)}")
    review["technical_pass"] = review["technical_pass"].map(parse_bool)
    for field in SCORE_FIELDS:
        review[field] = pd.to_numeric(review[field], errors="raise")
        invalid = ~review[field].isin([0, 1, 2])
        if invalid.any():
            raise ValueError(f"{field} must contain only integer scores 0, 1 or 2")
    if set(review["runtime"]) != EXPECTED_RUNTIMES:
        raise ValueError("Dense review must contain exactly Native Dense and RAG Dense")
    commits = {str(value) for value in review["commit"]}
    if len(commits) != 1 or len(next(iter(commits))) != 40:
        raise ValueError("Dense review must use one full shared commit SHA")
    review["video_total"] = review[SCORE_FIELDS].sum(axis=1)
    pair_rows = []
    for (prompt_id, category, prompt, seed), rows in review.groupby(
        ["prompt_id", "category", "prompt", "seed"], sort=False
    ):
        if set(rows["runtime"]) != EXPECTED_RUNTIMES or len(rows) != 2:
            raise RuntimeError(f"incomplete Dense runtime pair: {prompt_id} seed={seed}")
        pair_rows.append(
            {
                "prompt_id": prompt_id,
                "category": category,
                "prompt": prompt,
                "seed": int(seed),
                "technical_pass": bool(rows["technical_pass"].all()),
                "pair_min_total": float(rows["video_total"].min()),
                "native_dense_total": float(
                    rows.loc[rows["runtime"] == "native_dense", "video_total"].iloc[0]
                ),
                "rag_dense_total": float(
                    rows.loc[rows["runtime"] == "rag_dense", "video_total"].iloc[0]
                ),
            }
        )
    paired = pd.DataFrame(pair_rows)
    eligible = paired[paired["technical_pass"]]
    selected = []
    rankings = {}
    for category in CATEGORIES:
        group = eligible[eligible["category"] == category]
        seed_sets = group.groupby("prompt_id")["seed"].agg(
            lambda values: {int(value) for value in values}
        )
        qualified_ids = {
            prompt_id
            for prompt_id, seeds in seed_sets.items()
            if EXPECTED_SEEDS.issubset(seeds)
        }
        group = group[group["prompt_id"].isin(qualified_ids)]
        grouped = group.groupby(["prompt_id", "prompt", "category"], as_index=False).agg(
            seeds=("seed", "nunique"),
            two_seed_average=("pair_min_total", "mean"),
            worst_seed=("pair_min_total", "min"),
        )
        grouped = grouped[grouped["seeds"] >= 2].sort_values(
            ["two_seed_average", "worst_seed", "prompt_id"],
            ascending=[False, False, True],
        )
        if grouped.empty:
            raise RuntimeError(f"no two-seed Dense-qualified prompt for {category}")
        rankings[category] = grouped.to_dict(orient="records")
        selected.append(grouped.iloc[0].to_dict())
    payload = {
        "artifact_id": "dense_frozen_prompts_v1",
        "status": "frozen",
        "dense_screen_commit": next(iter(commits)),
        "selection_source": {
            "artifact_id": Path(args.review).name,
            "sha256": hashlib.sha256(Path(args.review).read_bytes()).hexdigest(),
        },
        "sparse_results_used": False,
        "required_seeds": sorted(EXPECTED_SEEDS),
        "score_fields": SCORE_FIELDS,
        "pair_rule": "minimum Native Dense/RAG Dense total for each prompt and seed",
        "ranking_rule": ["two_seed_average desc", "worst_seed desc", "prompt_id asc"],
        "category_rankings": rankings,
        "prompts": selected,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.prompt_output:
        Path(args.prompt_output).write_text(
            "\n".join(str(item["prompt"]) for item in selected) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
