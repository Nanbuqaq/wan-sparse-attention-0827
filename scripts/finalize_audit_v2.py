#!/usr/bin/env python3
"""Assemble the final Stage-2 method/matrix/evidence audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()

import pandas as pd
import av
import hashlib
import xml.etree.ElementTree as ET


REQUIRED_MAIN = {
    "block",
    "random",
    "local_3d",
    "fixed_k128",
    "qsort_local8",
    "token_oracle",
    "svg2",
    "adacluster",
    "svoo",
    "scope",
    "capacity_balanced",
    "radius_adaptive",
    "hierarchical",
    "product_quantized",
    "spatiotemporal",
    "query_metric",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_metadata(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(video=0))
        return {
            "path": str(path.relative_to(ROOT)),
            "frames": frames,
            "fps": float(stream.average_rate),
            "width": int(stream.width),
            "height": int(stream.height),
            "sha256": digest.hexdigest(),
        }


def junit_passed(path: Path, expected: int) -> bool:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(item.attrib.get("tests", 0)) for item in suites)
    failures = sum(int(item.attrib.get("failures", 0)) for item in suites)
    errors = sum(int(item.attrib.get("errors", 0)) for item in suites)
    return tests == expected and failures == 0 and errors == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-metrics", default="results/metrics/formal_stage2_v2")
    args = parser.parse_args()
    metrics = ROOT / args.formal_metrics
    cases = pd.read_csv(metrics / "case_metrics.csv")
    formal_audit = load_json(ROOT / "results/manifests/formal_stage2_v2/audit.json")
    correctness = load_json(ROOT / "results/metrics/correctness_v2/correctness_full.json")
    captured = load_json(ROOT / "results/metrics/captured_qkv_screen_v2.json")
    dense_screen = load_json(ROOT / "results/metrics/dense_prompt_screen_v2.json")
    numerical = load_json(ROOT / "results/metrics/correctness_v2/numerical_audit_summary.json")
    route_graph = load_json(ROOT / "results/manifests/formal_stage2_v2/route_graph_audit.json")
    same_route_kernel = load_json(ROOT / "results/metrics/same_route_kernel_benchmark_v2.json")
    curve = cases[cases["matrix_id"] == "density_curve_primary"]
    panel = pd.concat(
        [
            curve[curve["target_density"] == 0.25],
            cases[cases["matrix_id"] == "main_panel_d250_remaining"],
        ],
        ignore_index=True,
    )
    second = cases[cases["matrix_id"] == "second_seed_d250"]
    negative = cases[cases["matrix_id"] == "negative_holdout_d250"]
    kernel = cases[cases["matrix_id"] == "kernel_cross_backend_d250"]
    comparisons = [
        comparison_metadata(path)
        for path in sorted((ROOT / "results/videos/comparisons_v2").glob("*.mp4"))
    ]
    checks = {
        "route_and_backend_correctness_pass": correctness.get("status") == "pass",
        "numerical_formal_gate_open_with_disclosure": numerical.get("formal_quality_gate")
        in {"open", "open_with_non_equivalence_disclosure"},
        "strict_latent_status_preserved": numerical.get("strict_status") in {"pass", "fail"},
        "dynamic_backend_graph_divergence_audited": route_graph.get("status") in {
            "pass", "dynamic_graph_divergence_detected"
        },
        "strict_same_route_kernel_benchmark_pass": same_route_kernel.get("status") == "pass",
        "captured_qkv_screen_pass": captured.get("status") == "pass",
        "four_dense_prompts_frozen": len(dense_screen.get("accepted", [])) == 4,
        "formal_suite_audit_pass": formal_audit.get("status") == "pass",
        "density_curve_has_five_densities": sorted(curve["target_density"].unique().tolist()) == [0.05, 0.1, 0.15, 0.2, 0.25],
        "density_curve_has_all_methods": REQUIRED_MAIN.issubset(set(curve["base_method_id"])),
        "main_panel_four_prompts": panel["prompt_id"].nunique() == 4,
        "main_panel_has_all_methods": REQUIRED_MAIN.issubset(set(panel["base_method_id"])),
        "second_seed_has_all_methods": REQUIRED_MAIN.issubset(set(second["base_method_id"])),
        "two_negative_prompts": negative["prompt_id"].nunique() == 2,
        "negative_has_all_methods": REQUIRED_MAIN.issubset(set(negative["base_method_id"])),
        "kernel_variants_present": {
            "svg2_fixedgraph_native",
            "svg2_fixedgraph_csr",
            "svg2_varlen_native",
            "svg2_varlen_csr",
            "svoo_fixedgraph_native",
            "svoo_fixedgraph_csr",
            "svoo_varlen_native",
            "svoo_varlen_csr",
        }.issubset(set(kernel["base_method_id"])),
        "k256_recheck_present": bool((cases["base_method_id"] == "fixed_k256_negative").any()),
        "failed_calls_zero_in_main": int(panel["failed_calls"].sum()) == 0,
        "fallback_calls_zero_in_main": int(panel["fallback_calls"].sum()) == 0,
        "case_statistics_present": (metrics / "case_level_statistics.json").is_file(),
        "comparison_videos_complete": len(comparisons) >= 8
        and all(row["frames"] == 81 and abs(row["fps"] - 16.0) < 1e-6 for row in comparisons),
        "final_report_exists": (ROOT / "docs/FINAL_REPORT_V2.md").is_file(),
        "cpu_tests_3_passed": junit_passed(ROOT / "results/logs/cpu_tests_v2.xml", 3),
        "gpu_tests_22_passed": junit_passed(ROOT / "results/logs/gpu_tests_v2.xml", 22),
    }
    payload = {
        "schema_version": 2,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": {
            "case_rows": len(cases),
            "main_panel_cases": len(panel),
            "second_seed_cases": len(second),
            "negative_cases": len(negative),
            "kernel_cases": len(kernel),
        },
        "required_main_methods": sorted(REQUIRED_MAIN),
        "comparison_videos": comparisons,
    }
    output = ROOT / "results/manifests/final_audit_v2.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    public = ROOT / "results/manifests/public/final_audit_v2.json"
    public.parent.mkdir(parents=True, exist_ok=True)
    public.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
