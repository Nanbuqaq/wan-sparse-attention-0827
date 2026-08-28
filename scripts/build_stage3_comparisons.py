#!/usr/bin/env python3
"""Build and audit the fixed set of Stage-3 comparison videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()

import av


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="configs/stage3_formal_50step.json")
    args = parser.parse_args()
    suite = ROOT / args.suite
    methods = "dense,block,scope,svg2,coverage_cluster,vaware_cluster,stage3_hybrid"
    cases = [
        ("gymnast_ribbon", 9001, "main_gymnast"),
        ("skateboard_alley", 9001, "main_skateboard"),
        ("koi_reflections", 9001, "main_koi"),
        ("orchestra_conductor", 9001, "main_orchestra"),
        ("gymnast_ribbon", 65537, "second_seed_gymnast"),
        ("fox_snow", 9001, "negative_fox"),
        ("glassblower", 9001, "negative_glassblower"),
    ]
    output_dir = ROOT / "results/videos/stage3_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for prompt, seed, name in cases:
        output = output_dir / f"{name}.mp4"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/build_comparison_video.py"),
                "--suite", str(suite),
                "--matrix", "*",
                "--prompt", prompt,
                "--seed", str(seed),
                "--density", "0.25",
                "--methods", methods,
                "--output", str(output),
            ],
            check=True,
        )
        with av.open(str(output)) as container:
            stream = container.streams.video[0]
            frames = sum(1 for _ in container.decode(video=0))
            records.append(
                {
                    "prompt_id": prompt,
                    "seed": seed,
                    "path": str(output.relative_to(ROOT)),
                    "sha256": sha256(output),
                    "frames": frames,
                    "width": stream.width,
                    "height": stream.height,
                    "fps": float(stream.average_rate),
                    "status": "pass" if frames == 81 and float(stream.average_rate) == 16.0 else "fail",
                }
            )
    payload = {"schema_version": 3, "records": records, "status": "pass" if all(row["status"] == "pass" for row in records) else "fail"}
    audit = ROOT / "results/manifests/stage3/comparison_videos.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "videos": len(records), "audit": str(audit)}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
