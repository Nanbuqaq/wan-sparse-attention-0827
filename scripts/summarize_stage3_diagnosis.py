#!/usr/bin/env python3
"""Combine Stage-2 failures and Stage-3 captured V-objective evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap import ROOT, configure_runtime

configure_runtime()


SELF_METHODS = [
    "fixed_k128",
    "capacity_balanced",
    "radius_adaptive",
    "hierarchical",
    "product_quantized",
    "spatiotemporal",
    "query_metric",
]


def main() -> None:
    captured = json.loads((ROOT / "results/metrics/captured_qkv_screen_v2.json").read_text())
    captured_rows = [
        row
        for row in captured["rows"]
        if row.get("status") == "completed" and abs(float(row["density"]) - 0.25) < 1e-9
    ]
    screen = pd.DataFrame(
        {
            "method": row["method"],
            "output_relative_l2": row["output_error"]["relative_l2"],
            "attention_mass_recall": row["attention_mass_recall"],
            "route_ms": row["route_ms"],
        }
        for row in captured_rows
    )
    screen_summary = (
        screen.groupby("method")
        .agg(
            points=("method", "size"),
            output_relative_l2=("output_relative_l2", "mean"),
            attention_mass_recall=("attention_mass_recall", "mean"),
            route_ms=("route_ms", "mean"),
        )
        .reset_index()
    )
    correlation = float(
        np.corrcoef(screen["attention_mass_recall"], screen["output_relative_l2"])[0, 1]
    )

    formal = pd.read_csv(ROOT / "results/metrics/formal_stage2_v2/case_metrics.csv")
    main = formal[
        (formal["matrix_id"] == "main_panel_d250_remaining")
        | ((formal["matrix_id"] == "density_curve_primary") & (formal["target_density"].round(3) == 0.25))
    ]
    block = main[main["base_method_id"] == "block"].set_index("prompt_id")
    prompt_rows = []
    for method in SELF_METHODS:
        subset = main[main["base_method_id"] == method].set_index("prompt_id")
        for prompt_id in sorted(set(block.index) & set(subset.index)):
            prompt_rows.append(
                {
                    "method": method,
                    "prompt_id": prompt_id,
                    "psnr": float(subset.loc[prompt_id, "psnr_mean"]),
                    "block_psnr": float(block.loc[prompt_id, "psnr_mean"]),
                    "psnr_delta_vs_block": float(subset.loc[prompt_id, "psnr_mean"] - block.loc[prompt_id, "psnr_mean"]),
                    "ssim_delta_vs_block": float(subset.loc[prompt_id, "ssim_mean"] - block.loc[prompt_id, "ssim_mean"]),
                    "lpips_delta_vs_block": float(subset.loc[prompt_id, "lpips_mean"] - block.loc[prompt_id, "lpips_mean"]),
                }
            )
    prompt_table = pd.DataFrame(prompt_rows)
    collapse_summary = (
        prompt_table.groupby("method")
        .agg(
            prompts=("prompt_id", "size"),
            psnr_delta_mean=("psnr_delta_vs_block", "mean"),
            psnr_delta_worst=("psnr_delta_vs_block", "min"),
            prompts_below_block=("psnr_delta_vs_block", lambda values: int((values < 0).sum())),
        )
        .reset_index()
    )

    objectives = json.loads(
        (ROOT / "results/metrics/stage3_qkv_objectives_cpu/summary.json").read_text()
    )["objective_summary"]
    objective_lookup = {row["objective"]: row for row in objectives}
    qk = objective_lookup["qk_block"]["output_relative_l2_mean"]
    residual = objective_lookup["output_residual_oracle"]["output_relative_l2_mean"]
    prototype = objective_lookup["v_prototype"]["output_relative_l2_mean"]
    vnorm = objective_lookup["v_norm_only"]["output_relative_l2_mean"]
    payload = {
        "schema_version": 3,
        "status": "preliminary_pass_gpu_route_screen_pending",
        "stage2_captured_points": captured.get("points"),
        "attention_recall_vs_output_error_correlation": correlation,
        "captured_route_summary": screen_summary.to_dict(orient="records"),
        "formal_prompt_cluster_deltas": prompt_rows,
        "formal_cluster_summary": collapse_summary.to_dict(orient="records"),
        "v_objective_summary": objectives,
        "v_evidence": {
            "qk_block_output_relative_l2_mean": qk,
            "output_residual_oracle_output_relative_l2_mean": residual,
            "v_prototype_output_relative_l2_mean": prototype,
            "v_norm_only_output_relative_l2_mean": vnorm,
            "residual_improvement_vs_qk_fraction": (qk - residual) / qk,
            "prototype_improvement_vs_qk_fraction": (qk - prototype) / qk,
        },
        "interpretation": {
            "primary_cluster_failure": "shared route materialization removes original Block/local/time coverage after K-space grouping; seven different K-clustering variants fail on the same prompts, so K count or K-means family alone is not the explanation",
            "v_aware": "V norm alone is a poor retrieval target, while V prototypes and especially output-residual scoring reduce sampled-query output error at the same 25 percent block budget",
            "hard_region": "layer 0 is consistently the worst captured region across early, middle, and late denoise calls; later layers in the sampled capture are much easier",
            "next_gate": "validate stable-coverage and online residual routes on GPU, then use isolated 50-step calibration before freezing",
        },
    }
    output_dir = ROOT / "results/metrics/stage3_diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    screen_summary.to_csv(output_dir / "captured_route_summary.csv", index=False)
    prompt_table.to_csv(output_dir / "formal_prompt_cluster_deltas.csv", index=False)
    collapse_summary.to_csv(output_dir / "formal_cluster_summary.csv", index=False)

    lines = [
        "# Stage-3 preliminary clustering and V-aware diagnosis",
        "",
        "This document is generated from frozen Stage-2 evidence and Stage-3 sampled captured Q/K/V analysis. GPU validation of the new online routes is still a separate gate.",
        "",
        "## Why the basic clustering routes collapsed",
        "",
        "The seven K-clustering routes share one materialization pattern: K/V tokens are sorted by K-space labels, then a new fixed-block graph is selected without preserving Original-Block or local/time edges. Different cluster families therefore erase the same stability coverage. Their failures are strongly prompt-dependent, with the gymnast case collapsing across every family, while the conductor case is much less sensitive. This common pattern is stronger evidence than a K=128 versus K=256 explanation.",
        "",
        f"Across the existing captured screen, attention-mass recall and output relative-L2 have correlation {correlation:.3f}; recall by itself is not a sufficient output-quality target.",
        "",
        "## Does V matter?",
        "",
        f"At the same 25% block budget, sampled-query mean output relative-L2 is {qk:.4f} for QK block scoring, {prototype:.4f} for the V-prototype score, and {residual:.4f} for the offline output-residual oracle. The residual objective improves over QK by {(qk-residual)/qk*100:.1f}%. V norm alone is much worse ({vnorm:.4f}), so the useful signal is query-conditioned V contribution, not globally large V tokens.",
        "",
        "Layer 0 is the dominant worst region at early, middle, and late denoise calls in the sampled capture. The online Stage-3 route therefore protects Block/local coverage everywhere and permits V-aware scoring only in the remote remainder.",
        "",
        "## Evidence boundary",
        "",
        "The V residual result is an offline upper-bound diagnostic, not a completed online/video result. The deployable online approximation and all 50-step conclusions remain gated on GPU correctness and isolated calibration.",
    ]
    (ROOT / "docs/STAGE3_DIAGNOSIS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(output_dir), "report": str(ROOT / "docs/STAGE3_DIAGNOSIS.md")}, indent=2))


if __name__ == "__main__":
    main()
