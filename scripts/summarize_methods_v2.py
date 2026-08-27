#!/usr/bin/env python3
"""Create the complete Stage-2 method and kernel tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()

import pandas as pd


GROUPS = {
    "block": "baseline",
    "random": "baseline",
    "local_3d": "baseline",
    "fixed_k128": "baseline",
    "qsort_local8": "layout_baseline",
    "token_oracle": "oracle_baseline",
    "svg2": "paper",
    "adacluster": "paper",
    "svoo": "paper",
    "scope": "paper_derived",
    "capacity_balanced": "self_cluster",
    "radius_adaptive": "self_cluster",
    "hierarchical": "self_cluster",
    "product_quantized": "self_cluster",
    "spatiotemporal": "self_cluster",
    "query_metric": "self_cluster",
    "fixed_k256_negative": "negative_holdout",
    "svg2_fixedgraph_native": "kernel_variant",
    "svg2_fixedgraph_csr": "kernel_variant",
    "svg2_varlen_native": "kernel_variant",
    "svg2_varlen_csr": "kernel_variant",
    "svoo_fixedgraph_native": "kernel_variant",
    "svoo_fixedgraph_csr": "kernel_variant",
    "svoo_varlen_native": "kernel_variant",
    "svoo_varlen_csr": "kernel_variant",
}


def aggregate(table: pd.DataFrame) -> pd.DataFrame:
    return table.groupby("base_method_id").agg(
        cases=("prompt_id", "count"),
        prompts=("prompt_id", "nunique"),
        seeds=("seed", "nunique"),
        psnr=("psnr_mean", "mean"),
        ssim=("ssim_mean", "mean"),
        lpips=("lpips_mean", "mean"),
        flow_epe=("flow_epe_mean", "mean"),
        temporal_flicker=("temporal_flicker", "mean"),
        actual_density=("actual_density", "mean"),
        scheduled_density=("scheduled_density", "mean"),
        padding_ratio=("padding_ratio", "mean"),
        load_imbalance_cv=("load_imbalance_cv", "mean"),
        routing_p50_ms=("routing_p50_ms", "mean"),
        kernel_warm_p50_ms=("kernel_warm_p50_ms", "mean"),
        generation_elapsed_s=("generation_elapsed_s", "mean"),
        peak_memory_bytes=("peak_memory_bytes", "mean"),
        failed_calls=("failed_calls", "sum"),
        fallback_calls=("fallback_calls", "sum"),
    ).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-metrics", required=True)
    args = parser.parse_args()
    path = Path(args.case_metrics)
    if not path.is_absolute():
        path = ROOT / path
    table = pd.read_csv(path)
    curve = table[table["matrix_id"] == "density_curve_primary"]
    panel = pd.concat(
        [curve[curve["target_density"] == 0.25], table[table["matrix_id"] == "main_panel_d250_remaining"]],
        ignore_index=True,
    )
    second = table[table["matrix_id"] == "second_seed_d250"]
    negative = table[table["matrix_id"] == "negative_holdout_d250"]
    kernel = table[table["matrix_id"] == "kernel_cross_backend_d250"]
    tables = {
        "main_panel": aggregate(panel),
        "second_seed": aggregate(second),
        "negative_holdout": aggregate(negative),
        "kernel_variants": aggregate(kernel),
    }
    output_dir = path.parent
    payload = {"schema_version": 2, "tables": {}}
    for name, value in tables.items():
        value.insert(1, "method_group", value["base_method_id"].map(GROUPS).fillna("unknown"))
        value.to_csv(output_dir / f"{name}_method_table.csv", index=False)
        payload["tables"][name] = value.to_dict(orient="records")
    curve_table = curve.groupby(["base_method_id", "target_density"]).agg(
        psnr=("psnr_mean", "mean"),
        ssim=("ssim_mean", "mean"),
        lpips=("lpips_mean", "mean"),
        flow_epe=("flow_epe_mean", "mean"),
        temporal_flicker=("temporal_flicker", "mean"),
        actual_density=("actual_density", "mean"),
        scheduled_density=("scheduled_density", "mean"),
        generation_elapsed_s=("generation_elapsed_s", "mean"),
    ).reset_index()
    curve_table.to_csv(output_dir / "five_density_method_table.csv", index=False)
    payload["tables"]["five_density"] = curve_table.to_dict(orient="records")
    (output_dir / "complete_method_tables.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "main_methods": len(tables["main_panel"])}, indent=2))


if __name__ == "__main__":
    main()
