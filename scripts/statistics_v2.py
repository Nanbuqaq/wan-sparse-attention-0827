#!/usr/bin/env python3
"""Case-level paired statistics; frames are never treated as samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


METRICS = {
    "psnr_mean": 1.0,
    "ssim_mean": 1.0,
    "lpips_mean": -1.0,
    "flow_epe_mean": -1.0,
    "temporal_flicker": -1.0,
}


def bootstrap_ci(values: np.ndarray, *, seed: int, samples: int = 10000) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-metrics", required=True)
    parser.add_argument("--matrix", default="main_panel_d250")
    parser.add_argument("--references", default="block,fixed_k128")
    args = parser.parse_args()
    path = Path(args.case_metrics)
    if not path.is_absolute():
        path = ROOT / path
    table = pd.read_csv(path)
    table = table[table["matrix_id"] == args.matrix].copy()
    table = table[table["result_origin"].isin(["stage1_reused", "stage2_new"])]
    case_keys = ["prompt_id", "seed", "target_density"]
    methods = sorted(table["base_method_id"].unique())
    rows = []
    for reference in args.references.split(","):
        reference_table = table[table["base_method_id"] == reference]
        if reference_table.empty:
            continue
        local_rows = []
        for method in methods:
            if method == reference:
                continue
            candidate = table[table["base_method_id"] == method]
            paired = candidate.merge(reference_table, on=case_keys, suffixes=("_candidate", "_reference"))
            if paired.empty:
                continue
            row = {
                "reference": reference,
                "method": method,
                "cases": len(paired),
            }
            psnr_delta = paired["psnr_mean_candidate"].to_numpy() - paired["psnr_mean_reference"].to_numpy()
            row.update(
                {
                    "psnr_delta_mean": float(psnr_delta.mean()),
                    "psnr_ci_low": bootstrap_ci(psnr_delta, seed=42)[0],
                    "psnr_ci_high": bootstrap_ci(psnr_delta, seed=42)[1],
                    "wins": int((psnr_delta > 0.05).sum()),
                    "ties": int((np.abs(psnr_delta) <= 0.05).sum()),
                    "losses": int((psnr_delta < -0.05).sum()),
                    "worst_case_delta": float(psnr_delta.min()),
                    "worst_case": str(
                        paired.iloc[int(np.argmin(psnr_delta))]["prompt_id"]
                    )
                    + ":"
                    + str(paired.iloc[int(np.argmin(psnr_delta))]["seed"]),
                }
            )
            for metric, direction in METRICS.items():
                delta = direction * (
                    paired[f"{metric}_candidate"].to_numpy()
                    - paired[f"{metric}_reference"].to_numpy()
                )
                low, high = bootstrap_ci(delta, seed=43 + len(local_rows))
                row[f"{metric}_oriented_delta_mean"] = float(delta.mean())
                row[f"{metric}_oriented_ci_low"] = low
                row[f"{metric}_oriented_ci_high"] = high
            if len(psnr_delta) >= 2 and np.any(psnr_delta != 0):
                row["raw_p"] = float(wilcoxon(psnr_delta, zero_method="wilcox").pvalue)
            else:
                row["raw_p"] = 1.0
            local_rows.append(row)
        adjusted = holm_adjust([row["raw_p"] for row in local_rows])
        for row, value in zip(local_rows, adjusted):
            row["holm_p"] = value
            row["interpretation"] = (
                "evidence_of_difference" if value < 0.05 else "insufficient_evidence_to_distinguish"
            )
        rows.extend(local_rows)
    output_dir = path.parent
    pd.DataFrame(rows).to_csv(output_dir / "case_level_statistics.csv", index=False)
    payload = {
        "schema_version": 2,
        "statistical_unit": "complete_video_case",
        "frames_are_independent_samples": False,
        "bootstrap_samples": 10000,
        "holm_scope": "multiple method-comparison p-values",
        "rows": rows,
    }
    (output_dir / "case_level_statistics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()

