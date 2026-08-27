#!/usr/bin/env python3
"""Capture selected real Wan Q/K/V points while preserving Dense attention."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from bootstrap import ROOT, configure_runtime
from model_path import wan_model_path

configure_runtime()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from adapters.wan_sparse import _apply_rotary_emb, _project_qkv


LAYER_PATTERN = re.compile(r"(?:^|\.)blocks\.(\d+)\.attn1\.processor$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CaptureDenseProcessor:
    def __init__(self, *, layer: int, output_dir: Path, calls: set[int], heads: list[int]):
        self.layer = layer
        self.output_dir = output_dir
        self.calls = calls
        self.heads = heads
        self.call_index = 0
        self.records: list[dict] = []

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
        **_kwargs,
    ):
        if encoder_hidden_states is not None or attention_mask is not None:
            raise RuntimeError("capture processor only supports Wan self-attention")
        query, key, value = _project_qkv(attn, hidden_states)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        if self.call_index in self.calls:
            point = self.output_dir / f"layer_{self.layer:02d}_call_{self.call_index:03d}.pt"
            head_ids = torch.tensor(self.heads, device=q.device, dtype=torch.long)
            payload = {
                "q": q.index_select(1, head_ids).cpu(),
                "k": k.index_select(1, head_ids).cpu(),
                "v": v.index_select(1, head_ids).cpu(),
                "layer": self.layer,
                "call_index": self.call_index,
                "head_ids": self.heads,
                "shape_full": list(q.shape),
                "dtype": str(q.dtype),
            }
            torch.save(payload, point)
            self.records.append({"path": str(point), "sha256": sha256(point), **{k: payload[k] for k in ("layer", "call_index", "head_ids", "shape_full", "dtype")}})
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        output = output.transpose(1, 2).contiguous().flatten(2, 3).type_as(query)
        output = attn.to_out[0](output)
        self.call_index += 1
        return attn.to_out[1](output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-id", default="chef_motion")
    parser.add_argument(
        "--prompt",
        default="A documentary-style close-up of a chef rapidly chopping colorful vegetables in a busy restaurant kitchen, natural hand motion, steam and reflections, highly detailed, 4k",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--layers", default="0,9,19,29")
    parser.add_argument("--calls", default="1,49,99")
    parser.add_argument("--heads", default="0,3,6,9")
    args = parser.parse_args()
    layers = {int(value) for value in args.layers.split(",")}
    calls = {int(value) for value in args.calls.split(",")}
    heads = [int(value) for value in args.heads.split(",")]
    output_dir = ROOT / "results" / "captures" / "qkv_v2" / args.prompt_id / f"seed_{args.seed:06d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import WanPipeline
    from diffusers.utils import export_to_video

    model = str(wan_model_path())
    pipe = WanPipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
    if hasattr(pipe.scheduler, "shift"):
        pipe.scheduler.shift = 8.0
    processors = dict(pipe.transformer.attn_processors)
    capture_processors = []
    for name in list(processors):
        match = LAYER_PATTERN.search(name)
        if match is None:
            continue
        layer = int(match.group(1))
        if layer in layers:
            processor = CaptureDenseProcessor(
                layer=layer, output_dir=output_dir, calls=calls, heads=heads
            )
            processors[name] = processor
            capture_processors.append(processor)
    pipe.transformer.set_attn_processor(processors)
    pipe.enable_model_cpu_offload()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = pipe(
        prompt=args.prompt,
        height=480,
        width=832,
        num_frames=81,
        num_inference_steps=args.steps,
        guidance_scale=6.0,
        generator=generator,
        output_type="np",
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    video = output_dir / "dense_capture.mp4"
    export_to_video(result.frames[0], str(video), fps=16)
    records = [record for processor in capture_processors for record in processor.records]
    payload = {
        "status": "pass" if len(records) == len(layers) * len(calls) else "fail",
        "model": model,
        "prompt_id": args.prompt_id,
        "prompt": args.prompt,
        "seed": args.seed,
        "steps": args.steps,
        "layers": sorted(layers),
        "calls": sorted(calls),
        "heads": heads,
        "elapsed_s": elapsed,
        "records": records,
        "video": str(video),
        "video_sha256": sha256(video),
    }
    manifest = output_dir / "capture_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "records": len(records), "elapsed_s": elapsed, "manifest": str(manifest)}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
