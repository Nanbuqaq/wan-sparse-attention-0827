#!/usr/bin/env python3
"""Combine Dense-repeat, route, captured-Attention, and latent evidence."""

from __future__ import annotations

import json
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    correctness = load("results/metrics/correctness_v2/correctness_full.json")
    captured = load("results/metrics/correctness_v2/captured_100.json")
    latent = load("results/metrics/latent_equivalence_v2.json")
    direct_errors = [
        row["attention"]["relative_l2"]
        for row in correctness["backend_checks"] + captured["rows"]
    ]
    latent_errors = [row["latent"]["relative_l2"] for row in latent["methods"].values()]
    latent_spread = max(latent_errors) - min(latent_errors)
    checks = {
        "all_routes_and_backend_shapes_pass": correctness.get("status") == "pass",
        "captured_real_attention_pass": captured.get("status") == "pass",
        "dense_repeat_relative_l2_zero": latent["dense_repeat_noise"]["relative_l2"] == 0.0,
        "direct_attention_relative_l2_le_1pct": max(direct_errors) <= 0.01,
        "strict_latent_relative_l2_le_1pct": max(latent_errors) <= 0.01,
    }
    route_kernel_pass = all(
        value for key, value in checks.items() if key != "strict_latent_relative_l2_le_1pct"
    )
    route_independent_amplification = route_kernel_pass and latent_spread <= 5e-4
    if route_independent_amplification and not checks["strict_latent_relative_l2_le_1pct"]:
        classification = "route_and_kernel_attention_correct_but_multilayer_bf16_accumulation_exceeds_strict_latent_gate"
    elif all(checks.values()):
        classification = "strict_pass"
    else:
        classification = "core_correctness_failure"
    payload = {
        "schema_version": 2,
        "strict_status": "pass" if all(checks.values()) else "fail",
        "formal_quality_gate": (
            "open"
            if all(checks.values())
            else (
                "open_with_non_equivalence_disclosure"
                if route_independent_amplification
                else "blocked"
            )
        ),
        "classification": classification,
        "checks": checks,
        "max_direct_attention_relative_l2": max(direct_errors),
        "max_latent_relative_l2": max(latent_errors),
        "latent_relative_l2_spread_across_routes_backends": latent_spread,
        "route_independent_amplification": route_independent_amplification,
        "dense_repeat_noise": latent["dense_repeat_noise"],
        "note": "Strict latent equivalence remains failed. Formal quality ranking may open only because route/permutation/pair checks and direct Attention pass, and the latent offset is nearly identical with and without semantic permutation across all tested backends.",
    }
    output = ROOT / "results/metrics/correctness_v2/numerical_audit_summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
