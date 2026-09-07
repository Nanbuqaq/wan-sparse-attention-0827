#!/usr/bin/env python3
"""Source-weight gate audit and descriptive storyboards, not a paper ranking."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.offline_eval import output_error_metrics
from scripts.build_video_review_storyboards import analyze


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    paths = {"reference": args.case / "reference_longlive_rag.mp4", "tether": args.case / "tether_recovery_v3/tethermem.mp4",
             "neutral_split": args.case / "neutral_gate21/video.mp4"}
    cases, latents, errors = [], {}, {}
    for name, video in paths.items():
        run = json.loads(video.with_suffix(".run.json").read_text())
        if run["status"] != "completed" or run["seed"] != 20260912 or run["latent_frames"] != 21:
            raise ValueError("unmatched or incomplete gate video")
        latent_file = video.with_suffix(".latents.pt")
        latents[name] = torch.load(latent_file, map_location="cpu", weights_only=True)
        case = {"id": name, "method": name, "status": "pass", "prompt_id": "official_teapot", "seed": 20260912,
                "latent_frames": 21, "video": str(video), "video_sha256": sha(video), "latent_file_sha256": sha(latent_file)}
        cases.append(case)
        analyze(case, args.output, samples_per_quarter=16)
    for name in ("tether", "neutral_split"):
        errors[name] = {"latent_exact_reference": torch.equal(latents["reference"], latents[name]),
                        "all_latent": output_error_metrics(latents["reference"], latents[name]),
                        "last_chunk_latent": output_error_metrics(latents["reference"][:, -3:], latents[name][:, -3:])}
    result = {"status": "pass", "cases": cases, "latent_errors_not_absolute_quality": errors,
              "scope": "21_latent_81_pixel_source_weight_gate_not_long_video_conclusion",
              "backbone": "causal_forcing_plus_AE_no_LoRA", "old_LoRA_avgpool_results_not_pooled": True,
              "review_is_not_blind_human_preference": True, "initial_noise_SHA_not_recorded_in_all_early_gate_variants": True}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(errors, indent=2))


if __name__ == "__main__":
    main()
