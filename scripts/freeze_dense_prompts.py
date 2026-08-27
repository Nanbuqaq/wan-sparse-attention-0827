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
    required = {"prompt_id", "category", "seed", "native_dense_pass", "rag_dense_pass", "dense_quality_score", "prompt"}
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"missing review columns: {sorted(missing)}")
    review["native_dense_pass"] = review["native_dense_pass"].map(parse_bool)
    review["rag_dense_pass"] = review["rag_dense_pass"].map(parse_bool)
    review["dense_quality_score"] = pd.to_numeric(
        review["dense_quality_score"], errors="raise"
    )
    eligible = review[review["native_dense_pass"] & review["rag_dense_pass"]]
    selected = []
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
            seeds=("seed", "nunique"), score=("dense_quality_score", "mean")
        )
        grouped = grouped[grouped["seeds"] >= 2].sort_values(
            ["score", "prompt_id"], ascending=[False, True]
        )
        if grouped.empty:
            raise RuntimeError(f"no two-seed Dense-qualified prompt for {category}")
        selected.append(grouped.iloc[0].to_dict())
    payload = {
        "status": "frozen",
        "selection_source": str(Path(args.review).resolve()),
        "selection_source_sha256": hashlib.sha256(
            Path(args.review).read_bytes()
        ).hexdigest(),
        "sparse_results_used": False,
        "required_seeds": sorted(EXPECTED_SEEDS),
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
