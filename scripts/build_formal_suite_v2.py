#!/usr/bin/env python3
"""Build the immutable Stage-2 formal suite from frozen methods/prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="configs/frozen_methods_v2.json")
    parser.add_argument("--prompts", default="configs/formal_prompts_v2.json")
    parser.add_argument("--output", default="configs/formal_stage2_v2.json")
    args = parser.parse_args()
    frozen = json.loads((ROOT / args.methods).read_text(encoding="utf-8"))
    prompt_config = json.loads((ROOT / args.prompts).read_text(encoding="utf-8"))
    paper_self = [dict(item) for item in frozen["methods"]]
    required = {
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
    if {item["id"] for item in paper_self} != required:
        raise RuntimeError("frozen paper/self method set is incomplete")
    baselines = [
        {"id": "dense", "mode": "dense", "result_origin": "stage1_reused"},
        {"id": "block", "mode": "sparse", "method": "original_block", "backend": "fixed64_bf16", "parameter_origin": "stage2_cleanroom_fixed64", "result_origin": "stage2_new"},
        {"id": "random", "mode": "sparse", "method": "random_block", "backend": "fixed64_bf16", "parameter_origin": "exact_edge_budget", "result_origin": "stage2_new"},
        {"id": "local_3d", "mode": "sparse", "method": "local_3d", "backend": "fixed64_bf16", "parameter_origin": "exact_edge_budget", "route_params": {"frames_latent": 21, "height_latent": 30, "width_latent": 52}, "result_origin": "stage2_new"},
        {"id": "fixed_k128", "mode": "sparse", "method": "fixed_k128", "backend": "fixed64_bf16", "parameter_origin": "stage2_fixed_k_representative", "kmeans_init_iterations": 5, "kmeans_step_iterations": 1, "result_origin": "stage2_new"},
        {"id": "qsort_local8", "mode": "sparse", "method": "qsort_local8", "backend": "fixed64_bf16", "parameter_origin": "layout_sorting_baseline", "result_origin": "stage2_new"},
        {"id": "token_oracle", "mode": "sparse", "method": "token_oracle", "backend": "fixed64_bf16", "parameter_origin": "nondeployable_dense_qk_upper_bound", "result_origin": "stage2_new"},
        {"id": "fixed_k256_negative", "mode": "sparse", "method": "fixed_k256", "backend": "fixed64_bf16", "parameter_origin": "stage2_k256_negative_recheck", "kmeans_init_iterations": 5, "kmeans_step_iterations": 1, "result_origin": "negative_holdout"},
    ]
    for item in paper_self:
        if item["id"] in {"svg2", "svoo"}:
            item["route_params"] = {
                **item.get("route_params", {}),
                "record_route_graph_hash": True,
            }
    method_lookup = {item["id"]: item for item in paper_self}
    kernel_variants = []
    for method in ("svg2", "svoo"):
        base = method_lookup[method]
        for graph_kind in ("fixedgraph", "varlen"):
            for backend, suffix, backend_params in (
                ("varlen_triton_native", "native", {}),
                ("varlen_triton_csr", "csr", {"block_m": 64, "block_n": 32}),
            ):
                item = dict(base)
                route_params = dict(base.get("route_params", {}))
                route_params["record_route_graph_hash"] = True
                if graph_kind == "fixedgraph":
                    route_params["materialization"] = "fixed64_graph"
                else:
                    route_params.pop("materialization", None)
                item.update(
                    {
                        "id": f"{method}_{graph_kind}_{suffix}",
                        "backend": backend,
                        "backend_params": backend_params,
                        "route_params": route_params,
                        "parameter_origin": f"same_frozen_{graph_kind}_route_cross_backend",
                        "route_family": method,
                        "graph_kind": graph_kind,
                    }
                )
                kernel_variants.append(item)
    methods = baselines + paper_self + kernel_variants
    sparse_main_ids = [
        "block",
        "random",
        "local_3d",
        "fixed_k128",
        "qsort_local8",
        "token_oracle",
        *sorted(required),
    ]
    primary = prompt_config["primary_prompt_id"]
    formal = prompt_config["formal_prompt_ids"]
    negative = prompt_config["negative_prompt_ids"]
    suite = {
        "schema_version": 2,
        "freeze_status": "FROZEN_FORMAL_STAGE2_NO_RETUNING",
        "frozen_methods": args.methods,
        "frozen_prompts": args.prompts,
        "common": {
            "model": "${WAN_MODEL_PATH}",
            "height": 480,
            "width": 832,
            "frames": 81,
            "steps": 50,
            "guidance": 6.0,
            "shift": 8.0,
            "fps": 16,
        },
        "output_root": "results/videos/formal_stage2_v2",
        "manifest_root": "results/manifests/formal_stage2_v2",
        "methods": methods,
        "prompts": prompt_config["prompts"],
        "matrices": [
            {"id": "dense_reference_seed9001", "method_ids": ["dense"], "prompt_ids": [*formal, *negative], "seeds": [9001], "densities": [0.25]},
            {"id": "dense_reference_seed65537", "method_ids": ["dense"], "prompt_ids": [primary], "seeds": [65537], "densities": [0.25]},
            {"id": "density_curve_primary", "method_ids": sparse_main_ids, "prompt_ids": [primary], "seeds": [9001], "densities": [0.05, 0.10, 0.15, 0.20, 0.25]},
            {"id": "main_panel_d250_remaining", "method_ids": sparse_main_ids, "prompt_ids": [item for item in formal if item != primary], "seeds": [9001], "densities": [0.25]},
            {"id": "second_seed_d250", "method_ids": sparse_main_ids, "prompt_ids": [primary], "seeds": [65537], "densities": [0.25]},
            {"id": "negative_holdout_d250", "method_ids": sparse_main_ids, "prompt_ids": negative, "seeds": [9001], "densities": [0.25], "result_origin": "negative_holdout"},
            {"id": "kernel_cross_backend_d250", "method_ids": [item["id"] for item in kernel_variants], "prompt_ids": [primary, "koi_reflections"], "seeds": [9001], "densities": [0.25]},
            {"id": "k256_negative_recheck", "method_ids": ["fixed_k256_negative"], "prompt_ids": [primary], "seeds": [9001], "densities": [0.25], "result_origin": "negative_holdout"},
        ],
    }
    output = ROOT / args.output
    output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sparse_main_methods": len(sparse_main_ids)}, indent=2))


if __name__ == "__main__":
    main()
