#!/usr/bin/env python3
"""Aggregate and plot the strict same-RoutePlan kernel benchmark."""

from __future__ import annotations

import json

from bootstrap import ROOT, configure_runtime

configure_runtime()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    source = ROOT / "results/metrics/same_route_kernel_benchmark_v2.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    table = pd.DataFrame(payload["rows"])
    error_table = pd.json_normalize(table.pop("output_error_vs_native"))
    table = pd.concat((table, error_table.add_prefix("error_")), axis=1)
    summary = table.groupby(["method", "graph_kind", "backend"]).agg(
        points=("point", "count"),
        cold_kernel_ms=("cold_kernel_ms", "mean"),
        kernel_p50_ms=("kernel_p50_ms", "mean"),
        kernel_p90_ms=("kernel_p90_ms", "mean"),
        planner_p50_ms=("planner_p50_ms", "mean"),
        inverse_p50_ms=("inverse_p50_ms", "mean"),
        max_relative_l2=("error_relative_l2", "max"),
        min_cosine=("error_cosine", "min"),
    ).reset_index()
    native = summary[summary["backend"] == "varlen_triton_native"][["method", "graph_kind", "kernel_p50_ms"]].rename(columns={"kernel_p50_ms": "native_kernel_p50_ms"})
    summary = summary.merge(native, on=["method", "graph_kind"])
    summary["kernel_speedup_vs_native"] = summary["native_kernel_p50_ms"] / summary["kernel_p50_ms"]
    output_dir = ROOT / "results/metrics/formal_stage2_v2"
    summary.to_csv(output_dir / "same_route_kernel_summary.csv", index=False)
    figure_dir = ROOT / "results/figures/formal_stage2_v2"
    labels = [f"{row.method}-{row.graph_kind}-{row.backend.split('_')[-1]}" for row in summary.itertuples()]
    fig, axis = plt.subplots(figsize=(12, 5.5))
    axis.bar(labels, summary["kernel_p50_ms"])
    axis.set_ylabel("Warm kernel p50 (ms)")
    axis.set_title("Strict same-RoutePlan kernel comparison")
    axis.tick_params(axis="x", rotation=40)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "same_route_kernel_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

