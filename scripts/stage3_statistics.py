#!/usr/bin/env python3
"""Case-level Stage-3 summaries, confidence intervals, wins, and Pareto plot."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bootstrap import ROOT, configure_runtime

configure_runtime()


METRICS = {
    "psnr_mean": 1,
    "ssim_mean": 1,
    "lpips_mean": -1,
    "flow_epe_mean": -1,
    "temporal_flicker": -1,
}


def bootstrap(values: np.ndarray, seed: int = 3003, draws: int = 10000) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def exact_signflip_p(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistics.append(abs(float((values * np.asarray(signs)).mean())))
    return sum(value >= observed - 1e-12 for value in statistics) / len(statistics)


def holm_adjust(rows: list[dict], field: str, output_field: str) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: rows[index][field])
    running = 0.0
    count = len(rows)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * rows[index][field])
        running = max(running, adjusted)
        rows[index][output_field] = running


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-metrics", default="results/metrics/stage3_formal_50step/case_metrics.csv")
    parser.add_argument("--output-dir", default="results/metrics/stage3_formal_50step")
    args = parser.parse_args()
    table = pd.read_csv(ROOT / args.case_metrics)
    main = table[table["matrix_id"] == "stage3_main_d250"].copy()
    if main.empty:
        raise RuntimeError("missing Stage-3 main-panel cases")
    summary = (
        main.groupby("base_method_id")
        .agg(
            cases=("prompt_id", "size"),
            psnr_mean=("psnr_mean", "mean"),
            ssim_mean=("ssim_mean", "mean"),
            lpips_mean=("lpips_mean", "mean"),
            flow_epe_mean=("flow_epe_mean", "mean"),
            temporal_flicker=("temporal_flicker", "mean"),
            actual_density=("actual_density", "mean"),
            scheduled_density=("scheduled_density", "mean"),
            routing_p50_ms=("routing_p50_ms", "mean"),
            kernel_warm_p50_ms=("kernel_warm_p50_ms", "mean"),
            generation_elapsed_s=("generation_elapsed_s", "mean"),
            end_to_end_speedup_vs_dense=("end_to_end_speedup_vs_dense", "mean"),
            peak_memory_bytes=("peak_memory_bytes", "mean"),
        )
        .reset_index()
    )
    block = main[main["base_method_id"] == "block"].set_index(["prompt_id", "seed"])
    paired_rows = []
    for method in sorted(main["base_method_id"].unique()):
        if method == "block":
            continue
        candidate = main[main["base_method_id"] == method].set_index(["prompt_id", "seed"])
        keys = sorted(set(block.index) & set(candidate.index))
        if not keys:
            continue
        record = {"reference": "block", "method": method, "cases": len(keys)}
        for metric, direction in METRICS.items():
            delta = np.array(
                [direction * (float(candidate.loc[key, metric]) - float(block.loc[key, metric])) for key in keys],
                dtype=np.float64,
            )
            low, high = bootstrap(delta, seed=3003 + len(paired_rows) * 17)
            record[f"{metric}_oriented_delta_mean"] = float(delta.mean())
            record[f"{metric}_oriented_ci_low"] = low
            record[f"{metric}_oriented_ci_high"] = high
            if metric == "psnr_mean":
                record["psnr_case_level_p"] = exact_signflip_p(delta)
                record["wins"] = int((delta > 1e-6).sum())
                record["ties"] = int((np.abs(delta) <= 1e-6).sum())
                record["losses"] = int((delta < -1e-6).sum())
                record["worst_case_delta"] = float(delta.min())
        paired_rows.append(record)
    holm_adjust(paired_rows, "psnr_case_level_p", "psnr_holm_p")
    for record in paired_rows:
        record["interpretation"] = (
            "case_level_difference_detected_after_holm"
            if record["psnr_holm_p"] < 0.05
            else "insufficient_case_level_evidence_after_holm"
        )
    paired = pd.DataFrame(paired_rows)

    # A method is sparse-panel Pareto-optimal when no other method has both
    # higher PSNR and higher end-to-end speedup with one strict improvement.
    pareto = []
    for row in summary.itertuples():
        dominated = False
        for other in summary.itertuples():
            if other.base_method_id == row.base_method_id:
                continue
            if (
                other.psnr_mean >= row.psnr_mean
                and other.end_to_end_speedup_vs_dense >= row.end_to_end_speedup_vs_dense
                and (other.psnr_mean > row.psnr_mean or other.end_to_end_speedup_vs_dense > row.end_to_end_speedup_vs_dense)
            ):
                dominated = True
                break
        pareto.append(not dominated)
    summary["sparse_panel_pareto"] = pareto

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "stage3_method_table.csv", index=False)
    paired.to_csv(output_dir / "stage3_case_statistics_vs_block.csv", index=False)
    (output_dir / "stage3_case_statistics_vs_block.json").write_text(
        json.dumps(paired_rows, indent=2, sort_keys=True) + "\n"
    )

    figure_dir = ROOT / "results/figures/stage3_formal_50step"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for row in summary.itertuples():
        marker = "*" if row.sparse_panel_pareto else "o"
        size = 160 if row.sparse_panel_pareto else 80
        ax.scatter(row.end_to_end_speedup_vs_dense, row.psnr_mean, marker=marker, s=size)
        ax.annotate(row.base_method_id, (row.end_to_end_speedup_vs_dense, row.psnr_mean), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="Dense runtime")
    ax.set_xlabel("end-to-end speedup vs Dense")
    ax.set_ylabel("PSNR to matched Dense video (dB)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "quality_speed_pareto.png", dpi=200)
    plt.close(fig)
    print(json.dumps({"methods": len(summary), "paired_comparisons": len(paired), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
