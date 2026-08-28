#!/usr/bin/env python3
"""Generate the evidence-bounded Stage-3 final report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bootstrap import ROOT, configure_runtime

configure_runtime()


def markdown_table(table: pd.DataFrame, columns: list[str]) -> str:
    return table[columns].to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    metrics = ROOT / "results/metrics/stage3_formal_50step"
    methods = pd.read_csv(metrics / "stage3_method_table.csv").sort_values("psnr_mean", ascending=False)
    cases = pd.read_csv(metrics / "case_metrics.csv")
    paired = pd.read_csv(metrics / "stage3_case_statistics_vs_block.csv")
    diagnosis = json.loads((ROOT / "results/metrics/stage3_diagnosis/summary.json").read_text())
    qkv = json.loads((ROOT / "results/metrics/stage3_qkv_diagnostics/summary.json").read_text())
    kernel = json.loads((ROOT / "results/metrics/stage3_same_route_kernel.json").read_text())
    latent = json.loads((ROOT / "results/metrics/stage3_latent_100.json").read_text())
    backend100 = pd.read_csv(ROOT / "results/metrics/stage3_backend_100_50step/case_metrics.csv")
    review = json.loads((ROOT / "configs/stage3_formal_human_review.json").read_text())
    audit = json.loads((ROOT / "results/manifests/final_audit_stage3.json").read_text())
    hybrid = methods.set_index("base_method_id").loc["stage3_hybrid"]
    block = methods.set_index("base_method_id").loc["block"]
    scope = methods.set_index("base_method_id").loc["scope"]
    pair_hybrid = paired.set_index("method").loc["stage3_hybrid"]
    pair_coverage = paired.set_index("method").loc["coverage_cluster"]
    pair_vaware = paired.set_index("method").loc["vaware_cluster"]
    objective = {row["objective"]: row for row in qkv["objective_summary"]}
    route = {row["analysis_id"]: row for row in qkv["route_summary"]}
    fixed_kernel = next(row for row in kernel["rows"] if row["backend"] == "fixed64_bf16")
    best_csr = min((row for row in kernel["rows"] if row["backend"] == "varlen_triton_csr"), key=lambda row: row["combined_p50_ms"])
    second = cases[cases["matrix_id"] == "stage3_second_seed_d250"].sort_values("psnr_mean", ascending=False)
    negative = (
        cases[cases["matrix_id"] == "stage3_negative_d250"]
        .groupby("base_method_id")
        .agg(psnr_mean=("psnr_mean", "mean"), ssim_mean=("ssim_mean", "mean"), lpips_mean=("lpips_mean", "mean"), speedup=("end_to_end_speedup_vs_dense", "mean"))
        .reset_index()
        .sort_values("psnr_mean", ascending=False)
    )
    lines = [
        "# Wan short-video sparse attention Stage-3 final report",
        "",
        f"Audit status: **{audit['status']}** ({sum(audit['checks'].values())}/{len(audit['checks'])} checks).",
        "",
        "## Outcome",
        "",
        "Stage-3 produced a usable basic clustering route, an online V-aware route, and a final Block+local+cluster/V hybrid. All three preserve original token order and execute original Q/K/V. At 25% actual Q-K pair density, every Stage-3 method generated normal recognizable videos for all four formal prompts, the second seed, and both negative cases; none showed the subject-disappearance or large-white-region collapse of the cluster-only routes.",
        "",
        "## Four-prompt 25% main ranking",
        "",
        markdown_table(methods, ["base_method_id", "psnr_mean", "ssim_mean", "lpips_mean", "flow_epe_mean", "temporal_flicker", "routing_p50_ms", "kernel_warm_p50_ms", "generation_elapsed_s", "end_to_end_speedup_vs_dense", "actual_density"]),
        "",
        f"The final hybrid reaches {hybrid.psnr_mean:.3f} dB and {hybrid.end_to_end_speedup_vs_dense:.3f}x Dense speed. It is {hybrid.psnr_mean-block.psnr_mean:+.3f} dB above Block with 4 wins and 0 losses, while retaining a real speedup. It is only {hybrid.psnr_mean-scope.psnr_mean:+.3f} dB from SCOPE but is faster ({hybrid.end_to_end_speedup_vs_dense:.3f}x versus {scope.end_to_end_speedup_vs_dense:.3f}x).",
        "",
        f"The complete-case bootstrap CI for hybrid minus Block is [{pair_hybrid.psnr_mean_oriented_ci_low:+.3f}, {pair_hybrid.psnr_mean_oriented_ci_high:+.3f}] dB. The exact case-level sign-flip test has Holm-adjusted p={pair_hybrid.psnr_holm_p:.3f}; with four formal cases, this is reported as insufficient evidence for a significance claim, not equivalence.",
        "",
        "## Why the previous basic clustering collapsed",
        "",
        "Fixed-K and the six Stage-2 clustering families all sorted K/V by K-space labels and then selected a fresh fixed-block graph. They did not replace V with a centroid, but they removed the original Block/local/time connections. Seven different clustering families failed on the same prompts, so the shared materialization and missing coverage are more important than K=128 versus K=256 or a particular K-means variant.",
        "",
        f"On 12 captured Q/K/V points spanning layers 0/9/19/29 and early/middle/late denoise calls, Fixed-K128 output relative-L2 is {route['fixed_k128']['output_relative_l2_mean']:.3f}; the old clustering routes range from {min(route[name]['output_relative_l2_mean'] for name in ['spatiotemporal','radius_adaptive','hierarchical','capacity_balanced','query_metric','product_quantized']):.3f} to {max(route[name]['output_relative_l2_mean'] for name in ['spatiotemporal','radius_adaptive','hierarchical','capacity_balanced','query_metric','product_quantized']):.3f}. Block is {route['block']['output_relative_l2_mean']:.3f}, while stable coverage is {route['coverage_b80_l10']['output_relative_l2_mean']:.3f}. Layer 0 is the dominant worst region.",
        "",
        f"The frozen usable basic route reserves 70% of each row budget for Original-Block edges, 15% for local/time coverage, and 15% for remote cluster retrieval. In the four-prompt panel it is {pair_coverage.psnr_mean_oriented_delta_mean:+.3f} dB versus Block (3 wins, 1 loss) at {methods.set_index('base_method_id').loc['coverage_cluster','end_to_end_speedup_vs_dense']:.3f}x Dense speed.",
        "",
        "## What V contributes",
        "",
        f"At the same 25% block budget, sampled-query QK block scoring has output relative-L2 {objective['qk_block']['output_relative_l2_mean']:.4f}. V-prototype scoring reduces it to {objective['v_prototype']['output_relative_l2_mean']:.4f}; the offline output-residual oracle reaches {objective['output_residual_oracle']['output_relative_l2_mean']:.4f}. V norm alone is poor ({objective['v_norm_only']['output_relative_l2_mean']:.4f}). The useful signal is query-conditioned V contribution, not globally large V tokens.",
        "",
        f"The online prototype route is {pair_vaware.psnr_mean_oriented_delta_mean:+.3f} dB versus Block (3 wins, 1 loss) at {methods.set_index('base_method_id').loc['vaware_cluster','end_to_end_speedup_vs_dense']:.3f}x. The final hybrid uses the stronger residual approximation only for the remote remainder; it cannot evict Block/local guarantees.",
        "",
        "## 100% backend and latent separation",
        "",
        markdown_table(backend100.sort_values("base_method_id"), ["base_method_id", "psnr_mean", "ssim_mean", "lpips_mean", "flow_epe_mean", "temporal_flicker", "generation_elapsed_s", "end_to_end_speedup_vs_dense", "kernel_warm_p50_ms", "planner_p50_ms"]),
        "",
        f"Dense repeat latent noise is {latent['dense_repeat_noise']['relative_l2']:.6f}. At 100% density, fixed64 one-step latent relative-L2 is {latent['rows'][0]['latent_vs_dense']['relative_l2']:.4f} and Stage-3 fixed/CSR are {latent['rows'][1]['latent_vs_dense']['relative_l2']:.4f}/{latent['rows'][2]['latent_vs_dense']['relative_l2']:.4f}. All routes execute with exact 100% pairs and no fallback, but strict 1% latent equivalence fails. The 25% video error therefore contains a non-negligible multilayer BF16/backend component and cannot be attributed only to sparse retrieval.",
        "",
        "## Kernel decision",
        "",
        f"On one identical Stage-3 RoutePlan, fixed64 costs {fixed_kernel['combined_p50_ms']:.3f} ms. The best CSR setting costs {best_csr['combined_p50_ms']:.3f} ms including {best_csr['planner_p50_ms']:.3f} ms planning. CSR is retained as a correct negative result but is not the final backend. Low-frequency cluster refresh remains active, and original-order execution eliminates permutation/inverse overhead.",
        "",
        "## Robustness and absolute visual quality",
        "",
        "Second-seed results:",
        "",
        markdown_table(second, ["base_method_id", "psnr_mean", "ssim_mean", "lpips_mean", "flow_epe_mean", "temporal_flicker", "end_to_end_speedup_vs_dense"]),
        "",
        "Negative-case averages:",
        "",
        markdown_table(negative, ["base_method_id", "psnr_mean", "ssim_mean", "lpips_mean", "speedup"]),
        "",
        f"Formal visual review marks all {review['summary']['stage3_new_cases']} Stage-3 new-method cases as normal and collapse-free. SVG2 is numerically high in relative PSNR but visually collapsed in {review['summary']['svg2_visual_collapse']}/{review['summary']['svg2_cases']} reviewed cases. Relative-to-Dense metrics are therefore not treated as absolute video-quality scores.",
        "",
        "## LongLive migration",
        "",
        "The transferable design is: recent/local tokens receive exact guaranteed coverage; historical tokens are grouped only for retrieval; remote ranking uses cluster relevance plus a query-conditioned V/output-residual proxy; selected historical entries still execute their original KV. Refresh cluster metadata at low frequency, keep the total recent+history pair budget exact per call, and benchmark CSR only on the actual variable history graph. For fixed 64-token short-video graphs, CSR was slower; LongLive should not assume the result transfers to imbalanced history lengths.",
        "",
        "A practical LongLive starting allocation is 80% recent/content Block coverage, 10% explicit local/time coverage, and 10% history cluster/V-aware retrieval, with a small early-step shift toward the guaranteed recent budget. This is a migration hypothesis, not a completed LongLive experiment.",
        "",
        "## Evidence limits",
        "",
        "The main statistical unit is one complete prompt/seed video. Four main cases cannot support strong significance claims after multiple-comparison correction. PSNR/SSIM/LPIPS/Flow/flicker are fidelity metrics to matched Dense output; absolute visual review remains separate. The interrupted second-seed attempt is retained as a BrokenPipe failure and its successful rerun is the ranked result.",
        "",
        "## Artifacts",
        "",
        "- Frozen suite: `configs/stage3_formal_50step.json`",
        "- Final audit: `results/manifests/final_audit_stage3.json`",
        "- Case metrics and statistics: `results/metrics/stage3_formal_50step/`",
        "- Captured diagnostics: `results/metrics/stage3_qkv_diagnostics/`",
        "- Quality-speed figure: `results/figures/stage3_formal_50step/quality_speed_pareto.png`",
        "- Comparison videos: `results/videos/stage3_comparisons/`",
    ]
    output = ROOT / "docs/FINAL_REPORT_STAGE3.md"
    output.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "pass", "output": str(output), "lines": len(lines)}, indent=2))


if __name__ == "__main__":
    main()
