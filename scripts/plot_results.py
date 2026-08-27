#!/usr/bin/env python3
"""Create quality, timing, padding, and load-balance figures."""

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
    parser.add_argument("--metrics-dir", required=True)
    args = parser.parse_args()
    metrics_dir = Path(args.metrics_dir)
    if not metrics_dir.is_absolute():
        metrics_dir = ROOT / metrics_dir
    table = pd.read_csv(metrics_dir / "method_density_summary.csv")
    if table.empty:
        raise RuntimeError("summary table is empty")
    figure_dir = ROOT / "results" / "figures" / metrics_dir.name
    figure_dir.mkdir(parents=True, exist_ok=True)

    quality = [("psnr_mean", "PSNR vs Dense (dB)"), ("ssim_mean", "SSIM vs Dense"), ("lpips_mean", "LPIPS vs Dense")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for method, values in table.groupby("base_method_id"):
        values = values.sort_values("target_density")
        for axis, (column, label) in zip(axes, quality):
            axis.plot(values["actual_density"], values[column], marker="o", label=method)
            axis.set_xlabel("Actual logical Q-K pair density")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "quality_vs_density.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for method, values in table.groupby("base_method_id"):
        values = values.sort_values("target_density")
        axes[0].plot(values["actual_density"], values["generation_elapsed_s"], marker="o", label=method)
        axes[1].plot(values["actual_density"], values["scheduled_density"], marker="o", label=method)
        axes[2].plot(values["actual_density"], values["padding_ratio"], marker="o", label=method)
    axes[0].set_ylabel("Generation time (s)")
    axes[1].set_ylabel("Scheduled tile pairs / dense pairs")
    axes[2].set_ylabel("Padding ratio")
    for axis in axes:
        axis.set_xlabel("Actual logical Q-K pair density")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "runtime_padding_vs_density.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    maximum_density = table["target_density"].max()
    selected = table[table["target_density"] == maximum_density]
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    for row in selected.itertuples():
        axis.scatter(row.generation_elapsed_s, row.psnr_mean, s=70)
        axis.annotate(row.base_method_id, (row.generation_elapsed_s, row.psnr_mean), xytext=(5, 4), textcoords="offset points")
    axis.set_xlabel("Generation time (s)")
    axis.set_ylabel("PSNR vs Dense (dB)")
    axis.set_title(f"Quality-time at target density {maximum_density:.2f}")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "quality_time_max_density.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {figure_dir}")


if __name__ == "__main__":
    main()
