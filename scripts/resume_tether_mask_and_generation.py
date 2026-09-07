#!/usr/bin/env python3
"""Resume only missing official SAM2/Tether stages from a verified reference.

Keeps the successful reference/latent and all failed attempts. The two-pass
successful-workflow cost includes its original reference cost, not zero.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_tether_loading_variant import load_script


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sam2-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, source, out = args.base.resolve(), args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reference = base / "reference_longlive_rag.mp4"
    reference_record = json.loads((base / "reference_longlive_rag.loading_variant.json").read_text())
    if reference_record["status"] != "pass" or reference_record["retrieval_method_observed"] != "ae":
        raise ValueError("valid original-weight AE reference required")
    initial = json.loads((base / "loading_variant.json").read_text())["actual_commands"][0]["command"]
    def parameter(name):
        return initial[initial.index(name)+1]
    model = parameter("--model-root")
    prompt, seed, frames = parameter("--prompt"), parameter("--seed"), parameter("--frames")
    environment = os.environ.copy()
    environment["MODEL_ROOT"] = model
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    record = {"status": "running", "scope": "official_algorithm_with_deferred_text_loading_and_private_codec_dependencies",
        "reference_video_sha256": sha(reference), "reference": str(reference), "seed": int(seed), "latent_frames": int(frames),
        "reference_regenerated": False, "reference_successful_whole_wrapper_s": reference_record["whole_wrapper_wall_s"],
        "stages": []}
    sys.path.insert(0, str(source))
    original = load_script(source, "run_tethermem_pipeline.py")

    def run(label, command):
        started = time.perf_counter()
        with (out / (label + ".log")).open("x") as log:
            result = subprocess.run(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
        record["stages"].append({"stage": label, "returncode": result.returncode, "wall_s": time.perf_counter()-started, "command": command})
        (out / "progress.json").write_text(json.dumps(record, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{label} failed; successful earlier stages are retained")

    try:
        run("automatic_subject", [sys.executable, str(source / "scripts/auto_subject_box.py"), "--video", str(reference),
            "--sam2-repo", str(args.sam2_source), "--sam2-config", "configs/sam2/sam2_hiera_l.yaml",
            "--sam2-checkpoint", str(Path(model) / "sam2/sam2_hiera_large.pt"),
            "--output", str(out / "subject_box.json"), "--overlay", str(out / "auto_subject_overlay.png"),
            "--points-per-side", "16", "--points-per-batch", "64", "--device", "cuda"])
        box = json.loads((out / "subject_box.json").read_text())["selected"]["bbox_xyxy"]
        run("SAM2_tracking", [sys.executable, str(source / "scripts/extract_sam2_mask.py"), "--video", str(reference),
            "--box", *map(str, box), "--sam2-repo", str(args.sam2_source), "--sam2-config", "configs/sam2/sam2_hiera_l.yaml",
            "--sam2-checkpoint", str(Path(model) / "sam2/sam2_hiera_large.pt"), "--output", str(out / "subject_patch.npy"),
            "--raw-output", str(out / "subject_patch_raw.npy"), "--pixel-output", str(out / "subject_pixel_mask.npy"),
            "--metadata", str(out / "subject_mask.json"), "--overlay-dir", str(out / "sam2_overlays")])
        record["mask_quality"] = original._mask_quality_gate(SimpleNamespace(min_tracked_fraction=.95, max_held_fraction=.05), out / "subject_mask.json")
        run("Tether_generation", [sys.executable, str(ROOT / "scripts/run_tether_loading_variant.py"), "--source", str(source),
            "--script", "inference", "--", "--model-root", model, "--prompt", prompt, "--seed", seed, "--frames", frames,
            "--mode", "tethermem", "--mask", str(out / "subject_patch.npy"), "--output", str(out / "tethermem.mp4")])
        run("pair_diagnostics", [sys.executable, str(source / "scripts/evaluate_pair.py"), "--baseline", str(reference),
            "--tethermem", str(out / "tethermem.mp4"), "--output", str(out / "pair_diagnostics.json")])
        import av
        with av.open(str(out / "tethermem.mp4")) as container:
            decoded = sum(1 for _ in container.decode(video=0))
        if decoded != 4*int(frames)-3:
            raise ValueError("source decoder frame count mismatch; do not silently crop or pad")
        record["tether_video_sha256"] = sha(out / "tethermem.mp4")
        record["status"] = "pass"
    except BaseException as error:
        record["status"] = "fail"
        record["error"] = repr(error)
        raise
    finally:
        record["successful_workflow_cost_s"] = record["reference_successful_whole_wrapper_s"] + sum(s["wall_s"] for s in record["stages"] if s["returncode"] == 0)
        (out / "terminal.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
