#!/usr/bin/env python3
"""Build the frozen two-candidate 50-step calibration suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from screen_captured_qkv import candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--screen",
        default="results/metrics/captured_qkv_screen_v2.json",
    )
    parser.add_argument(
        "--output",
        default="configs/calibration_50step_v2.json",
    )
    parser.add_argument("--output-root", default="results/videos/calibration_50step_v2")
    parser.add_argument("--manifest-root", default="results/manifests/calibration_50step_v2")
    args = parser.parse_args()
    screen = json.loads((ROOT / args.screen).read_text(encoding="utf-8"))
    selected = set(screen["selected_two_per_method"])
    catalog = {
        f"{item['method']}:{item['candidate']}": item for item in candidates()
    }
    if selected - catalog.keys():
        raise RuntimeError(f"screen selected unknown candidates: {sorted(selected - catalog.keys())}")
    methods = [{"id": "dense", "mode": "dense", "result_origin": "stage2_new"}]
    method_ids = []
    for key in sorted(selected):
        item = catalog[key]
        method_id = f"{item['method']}__{item['candidate']}"
        method_ids.append(method_id)
        methods.append(
            {
                "id": method_id,
                "mode": "sparse",
                "method": item["method"],
                "backend": "fixed64_bf16",
                "parameter_origin": "captured_qkv_top2_candidate",
                "q_clusters": item["q_clusters"],
                "k_clusters": item["k_clusters"],
                "kmeans_init_iterations": item["init_iterations"],
                "kmeans_step_iterations": item["step_iterations"],
                "route_params": item["route_params"],
                "result_origin": "stage2_new",
            }
        )
    suite = {
        "schema_version": 2,
        "freeze_status": "FROZEN_BEFORE_50STEP_CALIBRATION",
        "source_screen": args.screen,
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
        "output_root": args.output_root,
        "manifest_root": args.manifest_root,
        "methods": methods,
        "prompts": [
            {
                "id": "chef_motion",
                "prompt": "A documentary-style close-up of a chef rapidly chopping colorful vegetables in a busy restaurant kitchen, natural hand motion, steam and reflections, highly detailed, 4k",
            },
            {
                "id": "neon_drone",
                "prompt": "An aerial drone shot weaving between futuristic skyscrapers at night during heavy rain, neon reflections, fast camera motion, cinematic, highly detailed, 4k",
            },
        ],
        "matrices": [
            {
                "id": "paper_and_self_parameter_calibration",
                "method_ids": ["dense", *method_ids],
                "prompt_ids": ["chef_motion", "neon_drone"],
                "seeds": [42],
                "densities": [0.10, 0.25],
            }
        ],
    }
    output = ROOT / args.output
    output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "methods": len(method_ids), "expected_tasks": 2 + len(method_ids) * 4}, indent=2))


if __name__ == "__main__":
    main()
