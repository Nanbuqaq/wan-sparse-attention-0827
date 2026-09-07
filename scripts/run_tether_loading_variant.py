#!/usr/bin/env python3
"""Explicit CPU-staged T5 loading variant of the unchanged official math.

The public constructor uploads FP32 T5 before pipeline.to(BF16), causing a
transient 24 GiB OOM. Suppress ONLY its constructor's single Module.cuda call;
the released DynamicSwapInstaller subsequently places the same BF16 weights.
This is NOT an unmodified-loader runtime result. Source files remain read-only.
"""
import argparse
import contextlib
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch


@contextlib.contextmanager
def defer_constructor_cuda():
    calls = []
    def keep_cpu(module, *args, **kwargs):
        calls.append(type(module).__name__)
        return module
    with patch.object(torch.nn.Module, "cuda", keep_cpu):
        yield calls


def load_script(source, name):
    path = source / "scripts" / name
    spec = importlib.util.spec_from_file_location("official_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--script", choices=("pipeline", "inference"), default="pipeline")
    parser.add_argument("forwarded", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = args.forwarded[1:] if args.forwarded[:1] == ["--"] else args.forwarded
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    os.chdir(source)
    output_flag = "--output-dir" if args.script == "pipeline" else "--output"
    output = Path(forwarded[forwarded.index(output_flag)+1]).resolve()
    root = output if args.script == "pipeline" else output.parent
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / ("loading_variant.json" if args.script == "pipeline" else output.stem + ".loading_variant.json")
    if record_path.exists():
        raise FileExistsError("do not overwrite a loading-variant run")
    record = {"status": "running", "variant": "defer_FP32_T5_cuda_until_BF16_dynamic_swap",
        "unmodified_loader": False, "source_files_modified": False, "weights_or_routing_modified": False,
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "actual_commands": [],
        "source_snapshot": json.loads((source / ".source_snapshot.json").read_text())}
    started = time.perf_counter()
    try:
        if args.script == "pipeline":
            official = load_script(source, "run_tethermem_pipeline.py")
            original_stage = official._stage
            def stage(label, command, **kwargs):
                command = list(command)
                if len(command) > 1 and Path(command[1]).name == "inference_tethermem.py":
                    command = [sys.executable, str(Path(__file__).resolve()), "--source", str(source), "--script", "inference", "--", *command[2:]]
                record["actual_commands"].append({"label": label, "command": command})
                record_path.write_text(json.dumps(record, indent=2) + "\n")
                return original_stage(label, command, **kwargs)
            official._stage = stage
        else:
            from utils.wan_wrapper import WanTextEncoder
            original_init = WanTextEncoder.__init__
            @functools.wraps(original_init)
            def cpu_init(self, *positional, **keywords):
                with defer_constructor_cuda() as calls:
                    original_init(self, *positional, **keywords)
                if len(calls) != 1 or any(p.device.type != "cpu" for p in self.text_encoder.parameters()):
                    raise RuntimeError("official text constructor changed; audit before adapting")
                record["deferred_cuda_calls"] = calls
                record["text_initial_device"] = "cpu"
            WanTextEncoder.__init__ = cpu_init
            from pipeline import CausalInferencePipeline
            original_inference = CausalInferencePipeline.inference
            def capture(self, *positional, **keywords):
                if any(p.dtype != torch.bfloat16 for p in self.text_encoder.parameters()):
                    raise RuntimeError("text weights must reach the same BF16 dtype before inference")
                result = original_inference(self, *positional, **keywords)
                if not keywords.get("return_latents"):
                    raise RuntimeError("official reproduction must retain the latent")
                video, latent = result
                if not torch.isfinite(latent).all() or not torch.isfinite(video).all():
                    raise ValueError("nonfinite official outputs")
                capture_started = time.perf_counter()
                torch.save(latent.detach().cpu(), output.with_suffix(".latents.pt"))
                record["artifact_capture_s"] = time.perf_counter() - capture_started
                record["latent_frames"] = int(latent.shape[1])
                record["retrieval_method_observed"] = self.compression_method
                if self.compression_method != "ae":
                    raise RuntimeError("refuse an average-pool retrieval fallback")
                return result
            CausalInferencePipeline.inference = capture
            official = load_script(source, "inference_tethermem.py")
        previous = sys.argv
        sys.argv = ["official_entry", *forwarded]
        try:
            code = official.main()
        finally:
            sys.argv = previous
        record["status"] = "pass" if code in (0, None) else "fail"
        return code
    except BaseException as error:
        record["status"] = "fail"
        record["error"] = repr(error)
        raise
    finally:
        record["whole_wrapper_wall_s"] = time.perf_counter() - started
        record_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
