#!/usr/bin/env python3
"""Compare saved attention/latent tensors for the 100% equivalence gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict) and len(value) == 1:
        candidate = next(iter(value.values()))
        if isinstance(candidate, torch.Tensor):
            return candidate
    raise TypeError(f"{path} does not contain one tensor")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--output")
    args = parser.parse_args()
    reference = load_tensor(Path(args.reference))
    candidate = load_tensor(Path(args.candidate))
    if reference.shape != candidate.shape:
        payload = {
            "status": "fail",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    else:
        delta = (reference.float() - candidate.float()).abs()
        payload = {
            "status": "pass" if torch.allclose(reference, candidate, atol=args.atol, rtol=args.rtol) else "fail",
            "shape": list(reference.shape),
            "dtype_reference": str(reference.dtype),
            "dtype_candidate": str(candidate.dtype),
            "max_abs": float(delta.max()) if delta.numel() else 0.0,
            "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
            "exact_equal": bool(torch.equal(reference, candidate)),
            "atol": args.atol,
            "rtol": args.rtol,
        }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

