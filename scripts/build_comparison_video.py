#!/usr/bin/env python3
"""Build a labelled horizontal comparison video from completed suite outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from bootstrap import ROOT, configure_runtime

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont
from run_matrix import expand_tasks, resolve_common


def ffmpeg_executable() -> str:
    value = shutil.which("ffmpeg")
    if value:
        return value
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--matrix", default="*", help="Matrix id or * to search across matrices")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--density", type=float, required=True)
    parser.add_argument("--methods", default="dense,block,fixed_k128,svg2_fixed,svg2_varlen")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    suite["common"] = resolve_common(suite["common"])
    records = []
    for expected in expand_tasks(suite):
        path = (ROOT / expected["output"]).with_suffix(".stats.json")
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        task = payload["task"]
        if (
            (args.matrix == "*" or task["matrix_id"] == args.matrix)
            and task["prompt_id"] == args.prompt
            and int(task["seed"]) == args.seed
        ):
            records.append(payload)
    by_method = {}
    for item in records:
        by_method.setdefault(item["task"]["base_method_id"], []).append(item)
    selected = []
    for method in args.methods.split(","):
        method = method.strip()
        candidates = by_method.get(method, [])
        if not candidates:
            continue
        if method != "dense":
            candidates = [
                item
                for item in candidates
                if abs(float(item["task"].get("density", -1)) - args.density) <= 1e-9
            ]
        if not candidates:
            continue
        candidates.sort(key=lambda item: item["task"]["matrix_id"])
        payload = candidates[0]
        selected.append((method, Path(payload["output"])))
    if len(selected) < 2:
        raise RuntimeError(f"need at least two completed methods, got {selected}")

    label_dir = ROOT / ".runtime" / "comparison_labels" / args.matrix / args.prompt
    label_dir.mkdir(parents=True, exist_ok=True)
    inputs = []
    filters = []
    for index, (method, video) in enumerate(selected):
        label = Image.new("RGB", (416, 38), color=(8, 12, 18))
        draw = ImageDraw.Draw(label)
        font = ImageFont.load_default(size=18)
        draw.text((10, 9), method, fill=(240, 245, 250), font=font)
        label_path = label_dir / f"{index:02d}_{method}.png"
        label.save(label_path)
        inputs.extend(["-i", str(video), "-loop", "1", "-i", str(label_path)])
        filters.append(
            f"[{2 * index}:v]scale=416:240,setsar=1[video{index}];"
            f"[{2 * index + 1}:v]scale=416:38[label{index}];"
            f"[video{index}][label{index}]overlay=0:0[cell{index}]"
        )
    filters.append(
        "".join(f"[cell{index}]" for index in range(len(selected)))
        + f"hstack=inputs={len(selected)}[out]"
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_executable(),
        "-y",
        "-loglevel",
        "error",
        *inputs,
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-frames:v",
        "81",
        "-r",
        "16",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(command, check=True)
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
