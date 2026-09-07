#!/usr/bin/env python3
"""One GPU per official prompt/seed lane, preserve failures and frame audits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]


def unpack(name, output):
    archive = ROOT / "vendor" / (name + ".tar.gz")
    lock = json.loads(archive.with_suffix(".lock.json").read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != lock["archive_sha256"]:
        raise ValueError("source archive SHA mismatch")
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            if not member.isfile() or Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ValueError("unsafe source archive entry")
        handle.extractall(output, filter="data")
    for item in lock["files"]:
        if hashlib.sha256((output / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("extracted source differs")
    return lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "configs/system/tether_official_reproduction_v1.json").read_text())
    cases = [(p, seed) for p in protocol["prompts"] for seed in protocol["long_reproduction"]["seeds"]]
    if args.gate_only:
        cases = [(protocol["prompts"][0], protocol["runtime_gate"]["seed"])]
    if args.lanes != len(cases) or not 0 <= args.lane < args.lanes:
        raise ValueError("this frozen batch has four distinct prompt/seed lanes")
    out = Path(os.environ["INFER_OUTPUT_DIR"])
    if args.validate_only:
        print(json.dumps({"cases": [(p["id"], s) for p, s in cases], "no_GPU_or_submission": True}))
        return
    shared = out / "prepared_sources"
    if args.prepare_only:
        shared.mkdir(parents=True, exist_ok=False)
        sources = {name: unpack(name, shared / name) for name in ("tethermem", "sam2")}
        (shared / "source_locks.json").write_text(json.dumps(sources, indent=2) + "\n")
        (out / "prepared.ok").write_text("exact source archives verified\n")
        return
    if not (out / "prepared.ok").is_file():
        raise RuntimeError("source preparation must finish before allocating GPUs")
    import torch
    import av
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    gpu = torch.cuda.get_device_name()
    if torch.cuda.get_device_properties(0).total_memory < 48 * 1024**3:
        raise RuntimeError("official unmodified pipeline requires a >=48 GiB GPU for this batch")
    prompt, seed = cases[args.lane]
    lane = out / f"lane{args.lane}"
    lane.mkdir(parents=True, exist_ok=False)
    model_root = Path(os.environ["INFER_WEIGHTS_DIR"]) / "models"
    source, sam2 = shared / "tethermem", shared / "sam2"
    environment = os.environ.copy()
    environment.update(MODEL_ROOT=str(model_root), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        PYTHONPATH=str(source) + ":" + str(Path(os.environ["INFER_WEIGHTS_DIR"]) / "python-extra")
            + ":" + str(Path(os.environ["INFER_WEIGHTS_DIR"]) / "python-overlay"))
    states = []
    # Only the short technical gate conditions the longer reproduction. There
    # is no quality-based parameter or prompt selection inside the lane.
    lengths = (21,) if args.gate_only else (21, protocol["long_reproduction"]["latent_frames"])
    for length in lengths:
        destination = lane / f"{prompt['id']}_s{seed}_lf{length}"
        command = [sys.executable, str(source / "scripts/run_tethermem_pipeline.py"), "--model-root", str(model_root),
            "--sam2-repo", str(sam2), "--prompt", prompt["text"], "--seed", str(seed), "--frames", str(length),
            "--output-dir", str(destination)]
        started = time.perf_counter()
        with (lane / f"lf{length}.log").open("x") as log:
            result = subprocess.run(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT)
        state = {"prompt_id": prompt["id"], "seed": seed, "latent_frames": length, "gpu": gpu,
                 "whole_pipeline_wall_s": time.perf_counter() - started, "exit_code": result.returncode,
                 "status": "pass" if result.returncode == 0 else "fail", "command": command}
        if result.returncode == 0:
            state["videos"] = []
            for name in ("reference_longlive_rag.mp4", "tethermem.mp4"):
                path = destination / name
                with av.open(str(path)) as container:
                    frames = sum(1 for _ in container.decode(video=0))
                state["videos"].append({"path": str(path), "frames": frames,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                if frames != 4 * length - 3:
                    state["status"] = "fail"
                    state["failure"] = "decoded frame count differs; never crop/pad silently"
        states.append(state)
        (lane / "terminal.json").write_text(json.dumps({"states": states, "status": state["status"]}, indent=2) + "\n")
        print(json.dumps(state), flush=True)
        if state["status"] != "pass":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
