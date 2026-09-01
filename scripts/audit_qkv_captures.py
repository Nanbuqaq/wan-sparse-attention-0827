#!/usr/bin/env python3
"""Audit a frozen layer/start QKV capture grid and emit reproducible evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ints(value: str) -> list[int]:
    output = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not output or len(output) != len(set(output)):
        raise ValueError(f"expected non-empty unique integer list: {value!r}")
    return output


def _as_int(value) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("capture scalar metadata must contain one value")
        return int(value.item())
    return int(value)


def audit_capture_grid(
    capture_dir: Path,
    *,
    layers: list[int],
    starts: list[int],
    query_tokens: int,
    heads: int,
    head_dim: int,
    frame_tokens: int,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    errors = []
    records = []
    expected_names = {
        f"layer{layer:02d}_start{start:08d}.pt" for layer in layers for start in starts
    }
    observed_names = {path.name for path in capture_dir.glob("*.pt")}
    missing = sorted(expected_names - observed_names)
    unexpected = sorted(observed_names - expected_names)
    if missing:
        errors.append(f"missing captures: {missing}")
    if unexpected:
        errors.append(f"unexpected captures: {unexpected}")

    for layer in layers:
        for start in starts:
            path = capture_dir / f"layer{layer:02d}_start{start:08d}.pt"
            if not path.is_file():
                continue
            record = {
                "path": str(path.resolve()),
                "artifact_id": path.name,
                "sha256": sha256(path),
                "expected_layer": layer,
                "expected_start": start,
            }
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                required = {"layer", "current_start", "query", "key", "value"}
                absent = sorted(required - set(payload))
                if absent:
                    raise ValueError(f"missing payload fields: {absent}")
                observed_layer = _as_int(payload["layer"])
                observed_start = _as_int(payload["current_start"])
                if observed_layer != layer or observed_start != start:
                    raise ValueError(
                        f"metadata mismatch layer/start={observed_layer}/{observed_start}"
                    )
                shapes = {}
                dtypes = {}
                finite = {}
                for name in ("query", "key", "value"):
                    tensor = payload[name]
                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError(f"{name} is not a tensor")
                    shapes[name] = list(tensor.shape)
                    dtypes[name] = str(tensor.dtype)
                    finite[name] = bool(torch.isfinite(tensor).all())
                query_shape = tuple(shapes["query"])
                key_shape = tuple(shapes["key"])
                value_shape = tuple(shapes["value"])
                expected_query_shape = (1, query_tokens, heads, head_dim)
                if query_shape != expected_query_shape:
                    raise ValueError(
                        f"query shape {query_shape} != {expected_query_shape}"
                    )
                if key_shape != value_shape:
                    raise ValueError(f"key/value shape mismatch: {key_shape} != {value_shape}")
                if (
                    len(key_shape) != 4
                    or key_shape[0] != 1
                    or key_shape[2:] != (heads, head_dim)
                    or key_shape[1] <= 0
                    or key_shape[1] % frame_tokens != 0
                ):
                    raise ValueError(f"invalid key/value shape: {key_shape}")
                if any(payload[name].dtype != dtype for name in ("query", "key", "value")):
                    raise ValueError(f"capture dtype mismatch: {dtypes}")
                if not all(finite.values()):
                    raise FloatingPointError(f"non-finite QKV tensor: {finite}")
                record.update(
                    {
                        "status": "pass",
                        "observed_layer": observed_layer,
                        "observed_start": observed_start,
                        "shapes": shapes,
                        "dtypes": dtypes,
                        "finite": finite,
                        "history_frames": key_shape[1] // frame_tokens,
                    }
                )
            except Exception as error:
                message = f"{path.name}: {type(error).__name__}: {error}"
                errors.append(message)
                record.update({"status": "fail", "error": message})
            records.append(record)
    return {
        "status": "pass" if not errors else "fail",
        "capture_dir": str(capture_dir.resolve()),
        "expected": {
            "layers": layers,
            "starts": starts,
            "captures": len(expected_names),
            "query_shape": [1, query_tokens, heads, head_dim],
            "key_value_frame_tokens": frame_tokens,
            "dtype": str(dtype),
        },
        "observed_captures": len(observed_names),
        "records": records,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--layers", default="0,9,19,29")
    parser.add_argument("--starts", default="28080,93600,177840")
    parser.add_argument("--query-tokens", type=int, default=4680)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--frame-tokens", type=int, default=1560)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = audit_capture_grid(
        Path(args.capture_dir),
        layers=parse_ints(args.layers),
        starts=parse_ints(args.starts),
        query_tokens=args.query_tokens,
        heads=args.heads,
        head_dim=args.head_dim,
        frame_tokens=args.frame_tokens,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "captures": len(payload["records"]), "errors": payload["errors"]}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
