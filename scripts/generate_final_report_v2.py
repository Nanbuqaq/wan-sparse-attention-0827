#!/usr/bin/env python3
"""Generate the concise Stage-2 final Markdown report from audited tables."""

from __future__ import annotations

import json
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()

import pandas as pd


def markdown_table(table: pd.DataFrame, columns: list[str]) -> str:
    value = table[columns].copy()
    for column in value.select_dtypes(include="number"):
        value[column] = value[column].map(lambda item: f"{item:.4f}" if pd.notna(item) else "")
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    rows = ["| " + " | ".join(str(item) for item in row) + " |" for row in value.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def main() -> None:
    metrics = ROOT / "results/metrics/formal_stage2_v2"
    audit = json.loads((ROOT / "results/manifests/final_audit_v2.json").read_text(encoding="utf-8"))
    numerical = json.loads((ROOT / "results/metrics/correctness_v2/numerical_audit_summary.json").read_text(encoding="utf-8"))
    prompts = json.loads((ROOT / "configs/formal_prompts_v2.json").read_text(encoding="utf-8"))
    k256 = json.loads((metrics / "k256_negative_recheck.json").read_text(encoding="utf-8"))
    main = pd.read_csv(metrics / "main_panel_method_table.csv").sort_values("psnr", ascending=False)
    kernel = pd.read_csv(metrics / "same_route_kernel_summary.csv").sort_values(["method", "graph_kind", "backend"])
    dynamic_kernel = pd.read_csv(metrics / "kernel_variants_method_table.csv").sort_values("kernel_warm_p50_ms")
    stats = pd.read_csv(metrics / "case_level_statistics.csv")
    second_seed = pd.read_csv(metrics / "second_seed_method_table.csv")
    main_lookup = main.set_index("base_method_id")
    second_lookup = second_seed.set_index("base_method_id")
    lines = [
        "# Wan short-video sparse attention Stage-2 report",
        "",
        f"Audit status: **{audit['status']}**.",
        "",
        "## Frozen evaluation design",
        "",
        f"Formal prompts: {', '.join(prompts['formal_prompt_ids'])}.",
        f"Negative holdouts: {', '.join(prompts['negative_prompt_ids'])}.",
        "All routing parameters were frozen using isolated 50-step calibration videos before the formal suite.",
        "",
        "## 25% multi-prompt method table",
        "",
        markdown_table(
            main,
            [
                "base_method_id",
                "method_group",
                "psnr",
                "ssim",
                "lpips",
                "flow_epe",
                "temporal_flicker",
                "routing_p50_ms",
                "kernel_warm_p50_ms",
                "generation_elapsed_s",
                "end_to_end_speedup_vs_dense",
                "actual_density",
                "scheduled_density",
            ],
        ),
        "",
        "Key observations:",
        "",
        f"- SVG2 has the highest four-prompt PSNR ({main_lookup.loc['svg2', 'psnr']:.3f} dB) but is slower than Dense ({main_lookup.loc['svg2', 'end_to_end_speedup_vs_dense']:.2f}x).",
        f"- SCOPE reaches {main_lookup.loc['scope', 'psnr']:.3f} dB at {main_lookup.loc['scope', 'end_to_end_speedup_vs_dense']:.2f}x and is the strongest paper-derived quality-speed compromise in the main panel.",
        f"- Original Block is the fastest strong baseline at {main_lookup.loc['block', 'end_to_end_speedup_vs_dense']:.2f}x with {main_lookup.loc['block', 'psnr']:.3f} dB.",
        "- None of the six required clean-room clustering families beats Block in the four-prompt main panel; all are retained as negative results.",
        f"- On seed 65537, SVOO and SVG2 are strongest ({second_lookup.loc['svoo', 'psnr']:.3f}/{second_lookup.loc['svg2', 'psnr']:.3f} dB), showing substantial seed sensitivity.",
        "",
        "## Variable-length kernel comparison",
        "",
        markdown_table(
            kernel,
            [
                "method",
                "graph_kind",
                "backend",
                "kernel_p50_ms",
                "kernel_p90_ms",
                "planner_p50_ms",
                "kernel_speedup_vs_native",
                "max_relative_l2",
            ],
        ),
        "",
        "Independent 50-step backend videos are retained as end-to-end evidence, but their route graphs diverge after layer 0 and they are not used for pure-kernel ranking.",
        markdown_table(
            dynamic_kernel,
            ["base_method_id", "psnr", "ssim", "lpips", "kernel_warm_p50_ms", "generation_elapsed_s", "scheduled_density", "padding_ratio"],
        ),
        "",
        "The first SVOO true-varlen CSR attempt failed because a zero-size padding Q cluster had no active edge. The planner was corrected to require edges only for non-empty Q clusters; the failure JSON is archived and the rerun completed successfully.",
        "",
        "## Correctness and numerical boundary",
        "",
        f"Strict numerical status: **{numerical['strict_status']}**; classification: `{numerical['classification']}`.",
        f"Maximum direct Attention relative L2: {numerical['max_direct_attention_relative_l2']:.6f}.",
        f"Maximum one-step latent relative L2: {numerical['max_latent_relative_l2']:.6f}.",
        "The report does not describe a failed strict latent gate as byte-level or numerical equivalence.",
        "",
        "## K256 negative recheck",
        "",
        f"K256 classification: **{k256['status']}**; Stage-2 PSNR delta versus K128: {k256['deltas_k256_minus_k128']['psnr_db']:.4f} dB.",
        k256["interpretation"],
        "The original Stage-1 collapse remains preserved independently; K128 stays in the required baseline table as an explicit negative result.",
        "",
        "## Case-level statistics",
        "",
        "Bootstrap and Holm calculations use complete prompt/seed videos as samples; the 81 frames are not independent observations.",
        markdown_table(
            stats.sort_values(["reference", "psnr_delta_mean"], ascending=[True, False]),
            ["reference", "method", "cases", "psnr_delta_mean", "psnr_ci_low", "psnr_ci_high", "wins", "ties", "losses", "worst_case_delta", "holm_p", "interpretation"],
        ),
        "",
        "## Evidence limits",
        "",
        "PSNR/SSIM/LPIPS/Flow/flicker measure fidelity to the matched Dense run, not absolute aesthetic quality. Dense prompts were therefore frozen using a separate normal-step visual review. Non-significant results are reported as insufficient evidence to distinguish methods, not as equivalence.",
    ]
    output = ROOT / "docs/FINAL_REPORT_V2.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
