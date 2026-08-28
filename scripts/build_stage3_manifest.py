#!/usr/bin/env python3
"""Build a complete local Stage-3 artifact manifest without publishing large files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    roots = [
        ROOT / "configs",
        ROOT / "adapters/routes",
        ROOT / "scripts",
        ROOT / "docs",
        ROOT / "results/videos/stage3_smoke_4step",
        ROOT / "results/videos/stage3_backend_100_50step",
        ROOT / "results/videos/stage3_calibration_50step",
        ROOT / "results/videos/stage3_formal_50step",
        ROOT / "results/videos/stage3_comparisons",
        ROOT / "results/metrics",
        ROOT / "results/figures",
        ROOT / "results/manifests/stage3",
        ROOT / "results/manifests/final_audit_stage3.json",
    ]
    files = []
    for root in roots:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            text = str(relative)
            if (
                (text.startswith("configs/") and "stage3" not in path.name)
                or (text.startswith("adapters/routes/") and path.name != "stage3.py")
                or (text.startswith("scripts/") and "stage3" not in path.name and path.name not in {"run_matrix.py", "run_on_free_gpu.py", "reuse_matching_tasks_between_suites.py"})
                or (text.startswith("docs/") and "STAGE3" not in path.name)
                or (text.startswith("results/metrics/") and "stage3" not in text)
                or (text.startswith("results/figures/") and "stage3" not in text)
            ):
                continue
            files.append(
                {
                    "path": text,
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                    "kind": "video" if path.suffix == ".mp4" else "metric" if "results/metrics" in text else "figure" if "results/figures" in text else "audit" if "results/manifests" in text else "source_or_config",
                }
            )
    files.sort(key=lambda item: item["path"])
    payload = {
        "schema_version": 3,
        "status": "complete",
        "files": files,
        "counts": {
            "files": len(files),
            "videos": sum(item["kind"] == "video" for item in files),
            "metrics": sum(item["kind"] == "metric" for item in files),
            "figures": sum(item["kind"] == "figure" for item in files),
            "audits": sum(item["kind"] == "audit" for item in files),
        },
    }
    output = ROOT / "results/manifests/stage3_manifest.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    sums = ROOT / "results/manifests/STAGE3_SHA256SUMS.txt"
    sums.write_text("".join(f"{item['sha256']}  {item['path']}\n" for item in files))
    print(json.dumps({"status": payload["status"], "counts": payload["counts"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
