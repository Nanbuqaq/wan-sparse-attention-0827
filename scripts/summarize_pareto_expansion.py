#!/usr/bin/env python3
"""Summarize frozen Pareto axes using complete-video statistical units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


AXES = (
    "density_curve",
    "formal_prompt_seed",
    "refresh_rope_factorial",
    "long_957",
)
VALID_VIDEO_STATUSES = {"pass", "negative"}
QUALITY_COLUMNS = ("ssim", "lpips", "late_ssim")


def nondominated(rows: pd.DataFrame) -> list[str]:
    maximize = ("ssim_mean", "late_ssim_mean")
    minimize = ("lpips_mean", "end_to_end_s_mean", "transfer_density_mean")
    selected = []
    for _, candidate in rows.iterrows():
        dominated = False
        for _, other in rows.iterrows():
            if candidate["method"] == other["method"]:
                continue
            no_worse = all(other[key] >= candidate[key] for key in maximize) and all(
                other[key] <= candidate[key] for key in minimize
            )
            strictly_better = any(
                other[key] > candidate[key] for key in maximize
            ) or any(other[key] < candidate[key] for key in minimize)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            selected.append(str(candidate["method"]))
    return sorted(selected)


def finite_mean(values: pd.Series) -> float | None:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if len(array) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--long-review")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cases_path = Path(args.cases).resolve()
    table = pd.read_csv(cases_path)
    if len(table) != 102 or table["case_id"].nunique() != 102:
        raise RuntimeError("frozen Pareto summary requires exactly 102 unique cases")
    table["axis_tags"] = table["expansion_axes"].fillna("[]").map(json.loads)
    baselines = table[table["method"] == "rag_dense"].copy()
    if len(baselines) != 12:
        raise RuntimeError("frozen Pareto summary requires 12 RAG Dense references")
    baseline_by = {
        (row.commit, row.prompt_id, int(row.seed), int(row.latent_frames)): row
        for row in baselines.itertuples(index=False)
    }

    manual = {}
    if args.long_review:
        review = pd.read_csv(args.long_review)
        if review["case_id"].duplicated().any():
            raise RuntimeError("long review contains duplicate case ids")
        manual = {row.case_id: row for row in review.itertuples(index=False)}

    paired_rows = []
    sparse = table[table["method"] != "rag_dense"]
    for row in sparse.itertuples(index=False):
        baseline = baseline_by.get(
            (row.commit, row.prompt_id, int(row.seed), int(row.latent_frames))
        )
        if baseline is None:
            raise RuntimeError(f"missing matched RAG Dense case: {row.case_id}")
        item = row._asdict()
        item.pop("axis_tags", None)
        item["dense_case_id"] = baseline.case_id
        item["dense_end_to_end_s"] = baseline.end_to_end_s
        item["dense_end_to_end_with_amortized_load_s"] = (
            baseline.end_to_end_with_amortized_load_s
        )
        item["speedup_vs_dense"] = (
            float(baseline.end_to_end_s) / float(row.end_to_end_s)
            if pd.notna(row.end_to_end_s) and float(row.end_to_end_s) > 0
            else np.nan
        )
        item["attention_speedup_vs_dense"] = (
            float(baseline.attention_s) / float(row.attention_s)
            if pd.notna(row.attention_s) and float(row.attention_s) > 0
            else np.nan
        )
        item["speedup_with_load_vs_dense"] = (
            float(baseline.end_to_end_with_amortized_load_s)
            / float(row.end_to_end_with_amortized_load_s)
            if pd.notna(row.end_to_end_with_amortized_load_s)
            and float(row.end_to_end_with_amortized_load_s) > 0
            else np.nan
        )
        item["transfer_reduction_vs_dense"] = (
            1.0 - float(row.h2d_bytes) / float(baseline.h2d_bytes)
            if pd.notna(row.h2d_bytes)
            and pd.notna(baseline.h2d_bytes)
            and float(baseline.h2d_bytes) > 0
            else np.nan
        )
        review = manual.get(row.case_id)
        item["visual_review_available"] = review is not None
        if review is not None:
            for field in (
                "subject_identity_1to5",
                "background_consistency_1to5",
                "irreversible_state_reset_count",
                "action_loop_count",
                "action_discontinuity_count",
                "freeze_count",
                "flicker_count",
                "camera_cut_count",
                "late_quarter_quality_1to5",
                "late_quarter_degradation_0to2",
                "reviewer",
                "review_notes",
            ):
                item[field] = getattr(review, field)
        paired_rows.append(item)
    paired = pd.DataFrame(paired_rows)

    exploded = []
    for _, row in paired.iterrows():
        tags = json.loads(row["expansion_axes"])
        for axis in tags:
            if axis in AXES:
                item = row.to_dict()
                item["axis"] = axis
                exploded.append(item)
    axis_cases = pd.DataFrame(exploded)
    if len(axis_cases) != 102:
        raise RuntimeError(
            f"expected 102 sparse axis memberships after overlap expansion, got {len(axis_cases)}"
        )

    group_columns = [
        "axis",
        "method",
        "latent_frames",
        "history_density",
        "rope_policy",
        "refresh_policy",
    ]
    summaries = []
    for keys, rows in axis_cases.groupby(group_columns, dropna=False):
        valid = rows[rows["status"].isin(VALID_VIDEO_STATUSES)]
        record = dict(zip(group_columns, keys))
        record.update(
            {
                "cases": len(rows),
                "pass_cases": int((rows["status"] == "pass").sum()),
                "negative_cases": int((rows["status"] == "negative").sum()),
                "fail_cases": int((rows["status"] == "fail").sum()),
                "ssim_mean": finite_mean(valid["ssim"]),
                "ssim_min": (
                    float(pd.to_numeric(valid["ssim"], errors="coerce").min())
                    if len(valid)
                    else None
                ),
                "lpips_mean": finite_mean(valid["lpips"]),
                "late_ssim_mean": finite_mean(valid["late_ssim"]),
                "end_to_end_s_mean": finite_mean(valid["end_to_end_s"]),
                "speedup_vs_dense_mean": finite_mean(valid["speedup_vs_dense"]),
                "speedup_with_load_vs_dense_mean": finite_mean(
                    valid["speedup_with_load_vs_dense"]
                ),
                "attention_speedup_vs_dense_mean": finite_mean(
                    valid["attention_speedup_vs_dense"]
                ),
                "transfer_density_mean": finite_mean(
                    valid["history_transfer_density"]
                ),
                "transfer_reduction_vs_dense_mean": finite_mean(
                    valid["transfer_reduction_vs_dense"]
                ),
            }
        )
        summaries.append(record)
    axis_summary = pd.DataFrame(summaries).sort_values(group_columns)

    method_rows = []
    for axis in ("formal_prompt_seed", "long_957"):
        rows = axis_cases[axis_cases["axis"] == axis]
        for method, method_cases in rows.groupby("method"):
            valid = method_cases[
                method_cases["status"].isin(VALID_VIDEO_STATUSES)
            ]
            method_rows.append(
                {
                    "axis": axis,
                    "method": method,
                    "cases": len(method_cases),
                    "pass_cases": int((method_cases["status"] == "pass").sum()),
                    "negative_cases": int(
                        (method_cases["status"] == "negative").sum()
                    ),
                    "fail_cases": int((method_cases["status"] == "fail").sum()),
                    "ssim_mean": finite_mean(valid["ssim"]),
                    "ssim_min": (
                        float(pd.to_numeric(valid["ssim"], errors="coerce").min())
                        if len(valid)
                        else None
                    ),
                    "lpips_mean": finite_mean(valid["lpips"]),
                    "late_ssim_mean": finite_mean(valid["late_ssim"]),
                    "end_to_end_s_mean": finite_mean(valid["end_to_end_s"]),
                    "speedup_vs_dense_mean": finite_mean(
                        valid["speedup_vs_dense"]
                    ),
                    "speedup_with_load_vs_dense_mean": finite_mean(
                        valid["speedup_with_load_vs_dense"]
                    ),
                    "attention_speedup_vs_dense_mean": finite_mean(
                        valid["attention_speedup_vs_dense"]
                    ),
                    "transfer_density_mean": finite_mean(
                        valid["history_transfer_density"]
                    ),
                    "transfer_reduction_vs_dense_mean": finite_mean(
                        valid["transfer_reduction_vs_dense"]
                    ),
                }
            )
    method_summary = pd.DataFrame(method_rows)

    formal = method_summary[method_summary["axis"] == "formal_prompt_seed"].copy()
    full_status = paired.groupby("method")["status"].value_counts().unstack(fill_value=0)
    eligible = []
    for method in formal["method"]:
        counts = full_status.loc[method]
        row = formal[formal["method"] == method].iloc[0]
        if (
            int(counts.get("fail", 0)) == 0
            and int(counts.get("negative", 0)) == 0
            and float(row["transfer_density_mean"]) <= 0.250001
        ):
            eligible.append(method)
    eligible_table = formal[formal["method"].isin(eligible)]
    pareto = nondominated(eligible_table) if len(eligible_table) else []

    negative_cases = paired[paired["status"].isin({"negative", "fail"})][
        [
            "case_id",
            "method",
            "prompt_id",
            "status",
            "negative_reasons",
            "rope_policy",
            "refresh_policy",
        ]
    ].to_dict(orient="records")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output / "paired_sparse_cases.csv", index=False)
    axis_summary.to_csv(output / "axis_summary.csv", index=False)
    method_summary.to_csv(output / "method_summary.csv", index=False)
    payload = {
        "status": "pass",
        "statistical_unit": "complete video",
        "timing_definitions": {
            "speedup_vs_dense": "generation and decode after model load",
            "speedup_with_load_vs_dense": "generation plus per-process amortized model load",
        },
        "case_metrics": str(cases_path),
        "cases": len(table),
        "sparse_cases": len(paired),
        "terminal_statuses": table["status"].value_counts().to_dict(),
        "system_eligible_methods": sorted(eligible),
        "system_pareto_methods": pareto,
        "negative_or_failed_cases": negative_cases,
        "interpretation": {
            "fixed_k256_history": "quality-oriented 25% transfer guardrail",
            "transfer_vaware_hybrid_history": "speed/transfer-oriented final candidate",
            "scope_ar": "quality diagnostic only; full transfer and runtime negatives exclude it from the system Pareto",
        },
        "manual_review_scope": (
            "16 long 957-frame videos; Codex sampled visual audit, not a blinded human panel"
            if args.long_review
            else "none"
        ),
    }
    (output / "pareto_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
