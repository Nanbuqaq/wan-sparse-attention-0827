#!/usr/bin/env python3
"""Freeze formal prompts using Dense-only two-seed review evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CATEGORIES = ["identity_scene", "irreversible_state", "human_action", "fast_motion"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    review = pd.read_csv(args.review)
    forbidden = [column for column in review.columns if "sparse" in column.lower()]
    if forbidden:
        raise ValueError(f"Dense prompt freeze cannot consume sparse columns: {forbidden}")
    required = {"prompt_id", "category", "seed", "native_dense_pass", "rag_dense_pass", "dense_quality_score", "prompt"}
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"missing review columns: {sorted(missing)}")
    eligible = review[
        review["native_dense_pass"].astype(bool) & review["rag_dense_pass"].astype(bool)
    ]
    selected = []
    for category in CATEGORIES:
        group = eligible[eligible["category"] == category]
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
        "sparse_results_used": False,
        "prompts": selected,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
