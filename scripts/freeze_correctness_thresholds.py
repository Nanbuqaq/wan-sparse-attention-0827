#!/usr/bin/env python3
"""Freeze kernel correctness thresholds from same-FA2 Dense repeat noise."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FLOORS = {
    "same_fa2_max_abs": 1e-5,
    "same_fa2_relative_l2": 1e-5,
    "different_kernel_max_abs": 2e-2,
    "different_kernel_relative_l2": 1e-2,
    "different_kernel_cosine_distance": 1e-3,
    "different_kernel_latent_relative_l2": 1e-2,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-repeat", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repeat = json.loads(Path(args.dense_repeat).read_text(encoding="utf-8"))
    mapping = {
        "same_fa2_max_abs": repeat["attention_max_abs"],
        "same_fa2_relative_l2": repeat["attention_relative_l2"],
        "different_kernel_max_abs": repeat["attention_max_abs"],
        "different_kernel_relative_l2": repeat["attention_relative_l2"],
        "different_kernel_cosine_distance": repeat["attention_cosine_distance"],
        "different_kernel_latent_relative_l2": repeat["latent_relative_l2"],
    }
    thresholds = {
        name: max(FLOORS[name], 5.0 * float(mapping[name])) for name in FLOORS
    }
    payload = {
        "status": "frozen",
        "source": str(Path(args.dense_repeat).resolve()),
        "rule": "max(fixed_floor, 5 * same-machine Dense repeat error)",
        "floors": FLOORS,
        "thresholds": thresholds,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

