#!/usr/bin/env python3
"""Classify the preregistered Stage-2 K256 recheck without expanding it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-case-metrics", required=True)
    args = parser.parse_args()
    path = Path(args.stage2_case_metrics)
    if not path.is_absolute():
        path = ROOT / path
    stage2 = pd.read_csv(path)
    k128 = stage2[
        (stage2["base_method_id"] == "fixed_k128")
        & (stage2["prompt_id"] == "gymnast_ribbon")
        & (stage2["seed"] == 9001)
        & (stage2["target_density"] == 0.25)
    ].iloc[0]
    k256 = stage2[stage2["base_method_id"] == "fixed_k256_negative"].iloc[0]
    old = pd.read_csv(ROOT / "results/metrics/formal_50step.final/case_metrics.csv")
    old_k256 = old[old["base_method_id"] == "fixed_k256"].iloc[0]
    deltas = {
        "psnr_db": float(k256.psnr_mean - k128.psnr_mean),
        "ssim": float(k256.ssim_mean - k128.ssim_mean),
        "lpips": float(k256.lpips_mean - k128.lpips_mean),
        "flow_epe": float(k256.flow_epe_mean - k128.flow_epe_mean),
        "temporal_flicker": float(k256.temporal_flicker - k128.temporal_flicker),
    }
    clearly_degraded = deltas["psnr_db"] <= -1.0 or deltas["lpips"] >= 0.05
    payload = {
        "status": "negative_holdout" if clearly_degraded else "recheck_not_clearly_degraded",
        "stage1": {
            "psnr": float(old_k256.psnr_mean),
            "lpips": float(old_k256.lpips_mean),
            "video": old_k256.video_path,
        },
        "stage2_k128": {"psnr": float(k128.psnr_mean), "lpips": float(k128.lpips_mean)},
        "stage2_k256": {"psnr": float(k256.psnr_mean), "lpips": float(k256.lpips_mean)},
        "deltas_k256_minus_k128": deltas,
        "automatic_degradation_rule": "PSNR <= -1 dB or LPIPS >= +0.05; manual collapse review may also mark negative",
        "expand_k256": False,
    }
    output = path.parent / "k256_negative_recheck.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

