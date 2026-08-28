#!/usr/bin/env python3
"""Video-level paired wins, worst cases, bootstrap CIs and frozen Pareto expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def hierarchical_bootstrap(rows: pd.DataFrame, column: str, *, resamples: int, seed: int = 20260827):
    rng = np.random.default_rng(seed)
    prompts = rows["prompt_id"].unique()
    values = []
    for _ in range(resamples):
        sampled_prompts = rng.choice(prompts, size=len(prompts), replace=True)
        sample = []
        for prompt in sampled_prompts:
            prompt_rows = rows[rows["prompt_id"] == prompt]
            indices = rng.integers(0, len(prompt_rows), size=len(prompt_rows))
            sample.extend(prompt_rows.iloc[indices][column].tolist())
        values.append(float(np.mean(sample)))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def nondominated(group: pd.DataFrame, maximize: list[str], minimize: list[str]) -> set[str]:
    selected = set()
    for _, candidate in group.iterrows():
        dominated = False
        for _, other in group.iterrows():
            if candidate["method"] == other["method"]:
                continue
            no_worse = all(other[name] >= candidate[name] for name in maximize) and all(
                other[name] <= candidate[name] for name in minimize
            )
            strictly = any(other[name] > candidate[name] for name in maximize) or any(
                other[name] < candidate[name] for name in minimize
            )
            if no_worse and strictly:
                dominated = True
                break
        if not dominated:
            selected.add(candidate["method"])
    return selected


def audit_eligibility(table: pd.DataFrame, rule: dict) -> tuple[set[str], list[dict]]:
    required_cases = int(rule["eligibility"]["required_terminal_basic_cases"])
    allowed = set(rule["eligibility"]["allowed_case_statuses"])
    baseline_methods = {"native_dense", "rag_dense"}
    eligible = set()
    audit = []
    for method, rows in table.groupby("method"):
        if method in baseline_methods:
            continue
        reasons = []
        if len(rows) != required_cases or rows["case_id"].nunique() != required_cases:
            reasons.append(f"requires exactly {required_cases} distinct basic cases")
        invalid_statuses = sorted(set(rows["status"]) - allowed)
        if invalid_statuses:
            reasons.append(f"technically invalid statuses: {invalid_statuses}")
        if rows["commit"].nunique() != 1:
            reasons.append("cases span multiple commits")
        if rows[["prompt_id", "seed"]].drop_duplicates().shape[0] != required_cases:
            reasons.append("basic prompt/seed pairs are incomplete or duplicated")
        for _, row in rows.iterrows():
            baselines = table[
                (table["method"] == row["baseline_method"])
                & (table["commit"] == row["commit"])
                & (table["prompt_id"] == row["prompt_id"])
                & (table["seed"] == row["seed"])
                & (table["latent_frames"] == row["latent_frames"])
            ]
            if len(baselines) != 1 or baselines.iloc[0]["status"] not in allowed:
                reasons.append(
                    f"missing technically valid same-commit baseline for {row['case_id']}"
                )
        audit.append({"method": method, "eligible": not reasons, "reasons": reasons})
        if not reasons:
            eligible.add(method)
    return eligible, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--rule", default="configs/pareto_rule.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.cases)
    rule = json.loads(Path(args.rule).read_text(encoding="utf-8"))
    required = {
        "case_id", "case_key_sha256", "commit", "method", "baseline_method",
        "routing_stage", "prompt_id", "seed", "latent_frames", "status",
        "psnr", "ssim", "lpips", "late_ssim", "end_to_end_s", "attention_s", "h2d_bytes",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"missing case columns: {sorted(missing)}")
    if table["case_id"].duplicated().any() or table["case_key_sha256"].duplicated().any():
        raise ValueError("case table contains duplicate identities")
    eligible_methods, eligibility_audit = audit_eligibility(table, rule)
    allowed = set(rule["eligibility"]["allowed_case_statuses"])
    eligible_rows = table[
        table["method"].isin(eligible_methods | {"native_dense", "rag_dense"})
        & table["status"].isin(allowed)
    ].copy()
    paired = []
    for _, row in eligible_rows.iterrows():
        if row["method"] not in eligible_methods:
            continue
        baseline = eligible_rows[
            (eligible_rows["method"] == row["baseline_method"])
            & (eligible_rows["commit"] == row["commit"])
            & (eligible_rows["prompt_id"] == row["prompt_id"])
            & (eligible_rows["seed"] == row["seed"])
            & (eligible_rows["latent_frames"] == row["latent_frames"])
        ]
        if len(baseline) != 1:
            raise RuntimeError(f"missing/duplicate baseline for {row['method']} {row['prompt_id']} {row['seed']}")
        base = baseline.iloc[0]
        item = row.to_dict()
        for metric in ("psnr", "ssim", "lpips", "late_ssim", "end_to_end_s", "attention_s", "h2d_bytes"):
            item[f"delta_{metric}"] = float(row[metric] - base[metric])
        paired.append(item)
    paired_table = pd.DataFrame(paired)
    if paired_table.empty:
        raise RuntimeError("no method has two technically valid paired basic cases")
    metric_columns = [
        "psnr",
        "ssim",
        "lpips",
        "late_ssim",
        "end_to_end_s",
        "attention_s",
        "h2d_bytes",
    ]
    if paired_table[metric_columns].isna().any().any():
        raise ValueError("Pareto metrics contain missing/NaN values")
    summaries = []
    for method, rows in paired_table.groupby("method"):
        ci_low, ci_high = hierarchical_bootstrap(
            rows, "delta_psnr", resamples=rule["bootstrap"]["resamples"]
        )
        summaries.append(
            {
                "method": method,
                "routing_stage": rows["routing_stage"].iloc[0],
                "cases": len(rows),
                "wins": int((rows["delta_psnr"] > 0).sum()),
                "losses": int((rows["delta_psnr"] < 0).sum()),
                "ties": int((rows["delta_psnr"] == 0).sum()),
                "delta_psnr_mean": rows["delta_psnr"].mean(),
                "delta_psnr_ci_low": ci_low,
                "delta_psnr_ci_high": ci_high,
                "worst_case_delta_psnr": rows["delta_psnr"].min(),
                "psnr_mean": rows["psnr"].mean(),
                "late_ssim_mean": rows["late_ssim"].mean(),
                "lpips_mean": rows["lpips"].mean(),
                "end_to_end_s_mean": rows["end_to_end_s"].mean(),
                "attention_s_mean": rows["attention_s"].mean(),
                "h2d_bytes_mean": rows["h2d_bytes"].mean(),
            }
        )
    summary = pd.DataFrame(summaries)
    expanded = set()
    for stage in rule["routing_groups"]:
        group = summary[summary["routing_stage"] == stage]
        if group.empty:
            continue
        expanded |= nondominated(group, rule["maximize"], rule["minimize"])
        expanded.add(group.loc[group["psnr_mean"].idxmax(), "method"])
        expanded.add(group.loc[group["end_to_end_s_mean"].idxmin(), "method"])
        expanded.add(group.loc[group["h2d_bytes_mean"].idxmin(), "method"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_table.to_csv(output_dir / "paired_video_cases.csv", index=False)
    summary.to_csv(output_dir / "method_video_summary.csv", index=False)
    (output_dir / "pareto_expansion.json").write_text(
        json.dumps(
            {
                "selected_methods": sorted(expanded),
                "eligible_methods": sorted(eligible_methods),
                "eligibility_audit": eligibility_audit,
                "rule": rule,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
