#!/usr/bin/env python3
"""Build prompt-level method-by-time contact sheets for formal manual review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


REVIEWABLE = {"pass", "negative"}
METHOD_ORDER = {
    "native_dense": 0,
    "native_block": 1,
    "rag_dense": 2,
    "rag_local": 3,
    "block64_history": 4,
    "random_history": 5,
    "token_oracle": 6,
    "kcluster32_history": 7,
    "fixed_k128_history": 8,
    "fixed_k256_history": 9,
    "radius_k256_ar": 10,
    "qmetric_k256_r32_ar": 11,
    "temporal_k256_t16_ar": 12,
    "sizesplit_k128_c2_ar": 13,
    "svoo_ar": 14,
    "scope_ar": 15,
    "coverage_cluster_history": 16,
    "vaware_cluster_history": 17,
    "transfer_vaware_hybrid_history": 18,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_selected(path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    output = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in indices:
                output[index] = frame.reformat(width=208, height=120).to_ndarray(
                    format="rgb24"
                )
    missing = sorted(indices - set(output))
    if missing:
        raise RuntimeError(f"missing selected frames in {path}: {missing}")
    return output


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    states_path = Path(args.states).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cases = [
        case
        for case in json.loads(states_path.read_text(encoding="utf-8"))["cases"]
        if case.get("status") in REVIEWABLE
    ]
    groups: dict[str, list[dict]] = {}
    for case in cases:
        groups.setdefault(str(case["prompt_id"]), []).append(case)
    records = []
    for prompt_id, prompt_cases in sorted(groups.items()):
        prompt_cases.sort(
            key=lambda case: (
                METHOD_ORDER.get(str(case["method"]), 999),
                str(case["method"]),
            )
        )
        frame_count = 4 * int(prompt_cases[0]["latent_frames"]) - 3
        indices = np.linspace(0, frame_count - 1, args.samples).round().astype(int)
        tile_width, tile_height, label_height = 208, 120, 30
        canvas = Image.new(
            "RGB",
            (args.samples * tile_width, len(prompt_cases) * (tile_height + label_height)),
            (4, 10, 18),
        )
        draw = ImageDraw.Draw(canvas)
        for row, case in enumerate(prompt_cases):
            video = Path(str(case["video"]))
            frames = decode_selected(video, set(int(index) for index in indices))
            y = row * (tile_height + label_height)
            label = f"{case['method']}  [{case.get('routing_stage', 'N/A')}]"
            draw.text((8, y + 6), label, fill=(238, 246, 255), font=font(14))
            for col, index in enumerate(indices):
                canvas.paste(
                    Image.fromarray(frames[int(index)]),
                    (col * tile_width, y + label_height),
                )
                if row == 0:
                    draw.text(
                        (col * tile_width + 5, y + label_height + 4),
                        f"f{int(index)}",
                        fill=(255, 215, 95),
                        font=font(11),
                    )
        output = output_root / f"{prompt_id}__methods.png"
        canvas.save(output)
        records.append(
            {
                "prompt_id": prompt_id,
                "methods": [case["method"] for case in prompt_cases],
                "frame_indices": indices.tolist(),
                "path": str(output),
                "sha256": sha256(output),
            }
        )
    manifest = {
        "status": "pass",
        "states": str(states_path),
        "states_sha256": sha256(states_path),
        "samples": args.samples,
        "records": records,
    }
    manifest_path = output_root / "contact_sheet_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"prompts": len(records), "output": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
