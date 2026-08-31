#!/usr/bin/env python3
"""Build prompt-level Native/RAG, two-seed contact sheets for Dense review."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw


RUNTIME_ORDER = {"native_dense": 0, "rag_dense": 1}


def sampled_frames(path: Path, count: int, width: int, height: int) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with av.open(str(path)) as container:
        decoded = [
            Image.fromarray(
                frame.reformat(width=width, height=height).to_ndarray(format="rgb24")
            )
            for frame in container.decode(video=0)
        ]
    if len(decoded) != 477:
        raise RuntimeError(f"{path}: decoded {len(decoded)} frames instead of 477")
    indices = np.linspace(0, len(decoded) - 1, count).round().astype(int)
    frames.extend(decoded[int(index)] for index in indices)
    return frames


def build_sheet(cases: list[dict], output: Path, *, samples: int) -> None:
    thumb_width, thumb_height = 208, 120
    columns = 8
    rows_per_case = (samples + columns - 1) // columns
    label_width = 260
    header_height = 42
    case_height = rows_per_case * thumb_height + 24
    canvas = Image.new(
        "RGB",
        (label_width + columns * thumb_width, header_height + len(cases) * case_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    prompt_id = cases[0]["prompt_id"]
    draw.text((12, 12), f"Dense review: {prompt_id}", fill="black")

    for case_index, case in enumerate(cases):
        top = header_height + case_index * case_height
        label = f"{case['runtime']}\nseed={case['seed']}"
        draw.multiline_text((12, top + 8), label, fill="black", spacing=6)
        frames = sampled_frames(
            Path(case["video"]), samples, thumb_width, thumb_height
        )
        for frame_index, image in enumerate(frames):
            x = label_width + (frame_index % columns) * thumb_width
            y = top + (frame_index // columns) * thumb_height
            canvas.paste(image, (x, y))
        draw.text(
            (label_width, top + rows_per_case * thumb_height + 4),
            "sampled uniformly from frame 0 to 476",
            fill="black",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args()

    payload = json.loads(Path(args.states).read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in payload["cases"]:
        if case.get("status") != "pass":
            continue
        grouped[str(case["prompt_id"])].append(case)

    output_dir = Path(args.output_dir)
    outputs = []
    for prompt_id, cases in sorted(grouped.items()):
        cases.sort(key=lambda case: (RUNTIME_ORDER[case["runtime"]], int(case["seed"])))
        if len(cases) != 4:
            raise RuntimeError(f"{prompt_id}: expected four Dense cases, got {len(cases)}")
        output = output_dir / f"{prompt_id}.png"
        build_sheet(cases, output, samples=args.samples)
        outputs.append(str(output))
    print(json.dumps({"prompts": len(outputs), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
