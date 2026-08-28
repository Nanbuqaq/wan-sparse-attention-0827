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
    parser.add_argument("--max-relative-l2", type=float)
    parser.add_argument("--min-cosine", type=float)
    parser.add_argument("--require-exact", action="store_true")
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
        reference_float = reference.float()
        candidate_float = candidate.float()
        signed_delta = candidate_float - reference_float
        delta = signed_delta.abs()
        relative_l2 = float(
            signed_delta.norm() / reference_float.norm().clamp_min(1e-12)
        )
        cosine = float(
            torch.nn.functional.cosine_similarity(
                reference_float.flatten(), candidate_float.flatten(), dim=0
            )
        )
        exact_equal = bool(torch.equal(reference, candidate))
        allclose = bool(
            torch.allclose(reference, candidate, atol=args.atol, rtol=args.rtol)
        )
        threshold_mode = (
            args.max_relative_l2 is not None
            or args.min_cosine is not None
            or args.require_exact
        )
        checks = {
            "finite": bool(
                torch.isfinite(reference_float).all()
                and torch.isfinite(candidate_float).all()
            ),
            "relative_l2": (
                True
                if args.max_relative_l2 is None
                else relative_l2 <= args.max_relative_l2
            ),
            "cosine": (
                True if args.min_cosine is None else cosine >= args.min_cosine
            ),
            "exact": True if not args.require_exact else exact_equal,
        }
        payload = {
            "status": (
                "pass"
                if (all(checks.values()) if threshold_mode else allclose)
                else "fail"
            ),
            "shape": list(reference.shape),
            "dtype_reference": str(reference.dtype),
            "dtype_candidate": str(candidate.dtype),
            "max_abs": float(delta.max()) if delta.numel() else 0.0,
            "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
            "relative_l2": relative_l2,
            "cosine": cosine,
            "cosine_distance": 1.0 - cosine,
            "exact_equal": exact_equal,
            "allclose": allclose,
            "atol": args.atol,
            "rtol": args.rtol,
            "max_relative_l2": args.max_relative_l2,
            "min_cosine": args.min_cosine,
            "require_exact": args.require_exact,
            "checks": checks,
        }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
