#!/usr/bin/env python3
"""Create compact PNG figures for the frozen LongLive Pareto expansion."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "fixed_k256_history": "#2563eb",
    "scope_ar": "#d97706",
    "transfer_vaware_hybrid_history": "#059669",
}
LABELS = {
    "fixed_k256_history": "Fixed-K256",
    "scope_ar": "SCOPE",
    "transfer_vaware_hybrid_history": "Final V-aware",
}


def tradeoff_plot(rows: pd.DataFrame, axis: str, output: Path) -> None:
    data = rows[rows["axis"] == axis]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for row in data.itertuples(index=False):
        color = COLORS[row.method]
        ax.scatter(
            row.speedup_with_load_vs_dense_mean,
            row.ssim_mean,
            s=90 + 120 * (1.0 - row.transfer_density_mean),
            color=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        ax.annotate(
            f"{LABELS[row.method]}\ntransfer={row.transfer_density_mean:.0%}",
            (row.speedup_with_load_vs_dense_mean, row.ssim_mean),
            xytext=(6, 7),
            textcoords="offset points",
            fontsize=9,
        )
    ax.axvline(1.0, color="#64748b", linestyle="--", linewidth=1)
    ax.set_xlabel("End-to-end speedup vs matched RAG Dense (including load)")
    ax.set_ylabel("Paired video SSIM (higher is closer to Dense)")
    ax.set_title(
        "Four prompts x two seeds" if axis == "formal_prompt_seed" else "957-frame long videos"
    )
    ax.grid(alpha=0.22)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def density_plot(rows: pd.DataFrame, output: Path) -> None:
    data = rows[rows["axis"] == "density_curve"].sort_values("history_density")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    for method, group in data.groupby("method"):
        label = LABELS[method]
        color = COLORS[method]
        density = 100 * group["history_density"]
        axes[0].plot(density, group["ssim_mean"], marker="o", color=color, label=label)
        axes[1].plot(
            density,
            group["speedup_with_load_vs_dense_mean"],
            marker="o",
            color=color,
        )
        axes[2].plot(
            density,
            100 * group["transfer_density_mean"],
            marker="o",
            color=color,
        )
    axes[0].set_ylabel("Paired video SSIM")
    axes[1].set_ylabel("Speedup incl. model load")
    axes[1].axhline(1.0, color="#64748b", linestyle="--", linewidth=1)
    axes[2].set_ylabel("History transfer density (%)")
    for ax in axes:
        ax.set_xlabel("History pair budget (%)")
        ax.grid(alpha=0.22)
    axes[0].legend(fontsize=8)
    fig.suptitle("Density sweep: quality, speed, and transfer are separate axes")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-summary", required=True)
    parser.add_argument("--axis-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    method_rows = pd.read_csv(args.method_summary)
    axis_rows = pd.read_csv(args.axis_summary)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tradeoff_plot(
        method_rows,
        "formal_prompt_seed",
        output / "formal_quality_speed_transfer.png",
    )
    tradeoff_plot(
        method_rows,
        "long_957",
        output / "long957_quality_speed_transfer.png",
    )
    density_plot(axis_rows, output / "density_quality_speed_transfer.png")
    print(f"wrote 3 PNG figures to {output}")


if __name__ == "__main__":
    main()
