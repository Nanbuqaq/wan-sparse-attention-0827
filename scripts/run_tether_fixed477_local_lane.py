#!/usr/bin/env python3
"""Frozen two-lane original-weight Tether study; independent cases, no retuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", type=int, choices=(0, 1), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sam2-source", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    protocol_path = ROOT / "configs/system/tether_runtime_fixed477_v2.json"
    protocol = json.loads(protocol_path.read_text())
    prompts = json.loads((ROOT / "configs/system/tether_official_reproduction_v1.json").read_text())["prompts"]
    prompt = prompts[args.lane]
    if args.validate_only:
        print(json.dumps({"lane": args.lane, "prompt": prompt, "seeds": protocol["seeds"], "latent_frames": 120,
                          "variants": protocol["variants_per_group"]}))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"protocol": protocol, "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                "prompt": prompt, "lane": args.lane, "source": str(args.source), "sam2_source": str(args.sam2_source),
                "execution_source_commit": os.environ.get("LONGLIVE_EXECUTION_COMMIT"),
                "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "wrapper_sha256": hashlib.sha256((ROOT / "scripts/run_tether_loading_variant.py").read_bytes()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    states = []
    for seed in protocol["seeds"]:
        case = args.output / f"{prompt['id']}_s{seed}"
        command = [sys.executable, str(ROOT / "scripts/run_tether_loading_variant.py"), "--source", str(args.source),
            "--release-unused-before-vae", "--continuous-vae", "--", "--model-root", str(args.models),
            "--sam2-repo", str(args.sam2_source), "--prompt", prompt["text"], "--frames", "120", "--seed", str(seed),
            "--output-dir", str(case)]
        started = time.perf_counter()
        with (args.output / f"seed{seed}.log").open("x") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        state = {"prompt_id": prompt["id"], "seed": seed, "status": "pass" if completed.returncode == 0 else "fail",
                 "pipeline_wall_s": time.perf_counter()-started, "pipeline_exit": completed.returncode, "neutral_status": "not_run"}
        if completed.returncode == 0:
            neutral = case / "neutral_split_sdpa"
            neutral.mkdir()
            command = [sys.executable, str(ROOT / "scripts/run_tether_loading_variant.py"), "--source", str(args.source),
                "--script", "inference", "--release-unused-before-vae", "--continuous-vae", "--neutral-routing", "--",
                "--model-root", str(args.models), "--prompt", prompt["text"], "--frames", "120", "--seed", str(seed),
                "--mode", "tethermem", "--mask", str(case / "subject_patch.npy"), "--output", str(neutral / "video.mp4")]
            with (neutral / "runner.log").open("x") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            state["neutral_status"] = "pass" if completed.returncode == 0 else "fail"
            if completed.returncode:
                state["status"] = "fail"
            else:
                import av
                videos = [case / "reference_longlive_rag.mp4", case / "tethermem.mp4", neutral / "video.mp4"]
                records, media = [], []
                for video in videos:
                    records.append(json.loads(video.with_name(video.stem + ".loading_variant.json").read_text()))
                    with av.open(str(video)) as container:
                        count = sum(1 for _ in container.decode(video=0))
                    media.append({"video": str(video), "frames": count, "sha256": hashlib.sha256(video.read_bytes()).hexdigest()})
                state["videos"] = media
                state["initial_noise_equal"] = len({r["initial_noise_sha256"] for r in records}) == 1
                state["all_latents_saved_before_VAE"] = all(r.get("latent_saved_before_VAE") for r in records)
                if not state["initial_noise_equal"] or not state["all_latents_saved_before_VAE"] or any(m["frames"] != 477 for m in media):
                    state["status"] = "fail"
                    state["failure"] = "paired noise/latent/frame audit failed"
        states.append(state)
        (args.output / "terminal.json").write_text(json.dumps({"status": "running" if len(states) < len(protocol["seeds"]) else
            ("pass" if all(s["status"] == "pass" for s in states) else "fail"), "states": states}, indent=2) + "\n")
        print(json.dumps(state), flush=True)
    if any(s["status"] != "pass" for s in states):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
