#!/usr/bin/env python3
"""Stage-2 density, main-panel, Pareto, and kernel figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-metrics", required=True)
    args = parser.parse_args()
    path = Path(args.case_metrics)
    if not path.is_absolute():
        path = ROOT / path
    table = pd.read_csv(path)
    figure_dir = ROOT / "results/figures/formal_stage2_v2"
    figure_dir.mkdir(parents=True, exist_ok=True)

    curve = table[table["matrix_id"] == "density_curve_primary"]
    metrics = [
        ("psnr_mean", "PSNR vs Dense (dB)"),
        ("ssim_mean", "SSIM vs Dense"),
        ("lpips_mean", "LPIPS vs Dense"),
        ("flow_epe_mean", "Flow EPE vs Dense"),
        ("temporal_flicker", "Temporal flicker delta"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
    for method, values in curve.groupby("base_method_id"):
        values = values.sort_values("actual_density")
        for axis, (column, label) in zip(axes, metrics):
            axis.plot(values["actual_density"], values[column], marker="o", linewidth=1, label=method)
            axis.set_xlabel("Actual Q-K pair density")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "density_curve_all_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    panel = pd.concat(
        [
            curve[curve["target_density"] == 0.25],
            table[table["matrix_id"] == "main_panel_d250_remaining"],
        ],
        ignore_index=True,
    )
    summary = panel.groupby("base_method_id").agg(
        cases=("prompt_id", "count"),
        psnr=("psnr_mean", "mean"),
        ssim=("ssim_mean", "mean"),
        lpips=("lpips_mean", "mean"),
        flow=("flow_epe_mean", "mean"),
        flicker=("temporal_flicker", "mean"),
        generation_s=("generation_elapsed_s", "mean"),
        speedup_vs_dense=("end_to_end_speedup_vs_dense", "mean"),
        routing_ms=("routing_p50_ms", "mean"),
        kernel_ms=("kernel_warm_p50_ms", "mean"),
        peak_memory_bytes=("peak_memory_bytes", "mean"),
        actual_density=("actual_density", "mean"),
        scheduled_density=("scheduled_density", "mean"),
        padding_ratio=("padding_ratio", "mean"),
    ).reset_index()
    summary.to_csv(figure_dir / "main_panel_summary.csv", index=False)

    fig, axis = plt.subplots(figsize=(9, 6.5))
    for row in summary.itertuples():
        axis.scatter(row.generation_s, row.psnr, s=55)
        axis.annotate(row.base_method_id, (row.generation_s, row.psnr), xytext=(4, 3), textcoords="offset points", fontsize=7)
    axis.set_xlabel("Generation time per 50-step video (s)")
    axis.set_ylabel("PSNR vs Dense (dB)")
    axis.set_title("25% multi-prompt quality-speed trade-off")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "quality_speed_pareto_d250.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    kernel = table[table["matrix_id"] == "kernel_cross_backend_d250"]
    canonical = panel[panel["base_method_id"].isin(["svg2", "svoo"])]
    kernel = pd.concat((canonical, kernel), ignore_index=True)
    kernel_summary = kernel.groupby("base_method_id").agg(
        kernel_ms=("kernel_warm_p50_ms", "mean"),
        planner_ms=("planner_p50_ms", "mean"),
        generation_s=("generation_elapsed_s", "mean"),
        scheduled_density=("scheduled_density", "mean"),
        padding_ratio=("padding_ratio", "mean"),
        load_cv=("load_imbalance_cv", "mean"),
    ).reset_index()
    kernel_summary.to_csv(figure_dir / "kernel_backend_summary.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(kernel_summary["base_method_id"], kernel_summary["kernel_ms"])
    axes[1].bar(kernel_summary["base_method_id"], kernel_summary["scheduled_density"])
    axes[2].bar(kernel_summary["base_method_id"], kernel_summary["padding_ratio"])
    axes[0].set_ylabel("Warm kernel p50 (ms)")
    axes[1].set_ylabel("Scheduled density")
    axes[2].set_ylabel("Padding ratio")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "kernel_backend_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {figure_dir}")


if __name__ == "__main__":
    main()
