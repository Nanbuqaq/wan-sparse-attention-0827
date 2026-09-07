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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def tensor_return_adapter(function):
    """Adapt an API return container, never change the computed tensor."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        result = function(*args, **kwargs)
        if isinstance(result, tuple):
            if not result or not isinstance(result[0], torch.Tensor):
                raise TypeError("unsupported FA3 return signature")
            return result[0]
        if not isinstance(result, torch.Tensor):
            raise TypeError("unsupported attention return signature")
        return result
    return wrapped


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
    parser.add_argument("--keep-original-text-loading", action="store_true")
    parser.add_argument("--fa3-return-compat", action="store_true")
    parser.add_argument("--release-unused-before-vae", action="store_true")
    parser.add_argument("--continuous-vae", action="store_true")
    parser.add_argument("--neutral-routing", action="store_true")
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
    record = {"status": "running", "variant": "recorded_official_runtime_compatibility",
        "unmodified_loader": args.keep_original_text_loading, "source_files_modified": False, "weights_or_routing_modified": False,
        "flags": {"deferred_text": not args.keep_original_text_loading, "fa3_return_container": args.fa3_return_compat,
                  "empty_cache_before_VAE": args.release_unused_before_vae, "continuous_VAE": args.continuous_vae,
                  "neutral_split_SDPA_control": args.neutral_routing},
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
                    flags = [flag for enabled, flag in ((args.keep_original_text_loading, "--keep-original-text-loading"),
                        (args.fa3_return_compat, "--fa3-return-compat"), (args.release_unused_before_vae, "--release-unused-before-vae"),
                        (args.continuous_vae, "--continuous-vae"), (args.neutral_routing, "--neutral-routing")) if enabled]
                    command = [sys.executable, str(Path(__file__).resolve()), "--source", str(source), "--script", "inference", *flags, "--", *command[2:]]
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
            if not args.keep_original_text_loading:
                WanTextEncoder.__init__ = cpu_init
            if args.fa3_return_compat:
                try:
                    import flash_attn_interface
                except ModuleNotFoundError:
                    record["fa3_package_present"] = False
                else:
                    record["fa3_package_present"] = True
                    record["fa3_interface_sha256"] = hashlib.sha256(Path(flash_attn_interface.__file__).read_bytes()).hexdigest()
                    flash_attn_interface.flash_attn_varlen_func = tensor_return_adapter(flash_attn_interface.flash_attn_varlen_func)
            from pipeline import CausalInferencePipeline
            original_inference = CausalInferencePipeline.inference
            def capture(self, *positional, **keywords):
                from adapters.longlive_sparse.history_cache import tensor_sha256
                noise = keywords.get("noise", positional[0] if positional else None)
                if noise is None:
                    raise ValueError("explicit initial noise required for paired reproduction")
                record["initial_noise_sha256"] = tensor_sha256(noise)
                if any(p.dtype != torch.bfloat16 for p in self.text_encoder.parameters()):
                    raise RuntimeError("text weights must reach the same BF16 dtype before inference")
                original_decode = self.vae.decode_to_pixel_chunk
                def checkpoint_decode(latent, *decode_args, **decode_kwargs):
                    if not torch.isfinite(latent).all():
                        raise ValueError("nonfinite latent before VAE")
                    from adapters.longlive_sparse.history_cache import tensor_sha256
                    capture_started = time.perf_counter()
                    torch.save(latent.detach().cpu(), output.with_suffix(".latents.pt"))
                    record["artifact_capture_before_VAE_s"] = time.perf_counter() - capture_started
                    record["latent_saved_before_VAE"] = True
                    record["latent_sha256"] = tensor_sha256(latent)
                    record["latent_frames"] = int(latent.shape[1])
                    record_path.write_text(json.dumps(record, indent=2) + "\n")
                    if args.release_unused_before_vae:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    if args.continuous_vae:
                        if decode_args or decode_kwargs.get("use_cache", False):
                            raise ValueError("continuous wrapper supports complete-video decode only")
                        from adapters.longlive_sparse.video_decode import decode_latents_chunked_exact
                        return decode_latents_chunked_exact(self.vae, latent, chunk_size=decode_kwargs.get("chunk_size", 60))
                    return original_decode(latent, *decode_args, **decode_kwargs)
                with patch.object(self.vae, "decode_to_pixel_chunk", checkpoint_decode):
                    result = original_inference(self, *positional, **keywords)
                if not keywords.get("return_latents"):
                    raise RuntimeError("official reproduction must retain the latent")
                video, latent = result
                if not torch.isfinite(latent).all() or not torch.isfinite(video).all():
                    raise ValueError("nonfinite official outputs")
                if not record.get("latent_saved_before_VAE"):
                    raise RuntimeError("decoder-boundary latent checkpoint missing")
                record["latent_frames"] = int(latent.shape[1])
                record["retrieval_method_observed"] = self.compression_method
                record["retrieval_trace"] = self.memory_indices_log
                if self.compression_method != "ae":
                    raise RuntimeError("refuse an average-pool retrieval fallback")
                return result
            CausalInferencePipeline.inference = capture
            official = load_script(source, "inference_tethermem.py")
            if args.neutral_routing:
                original_load = official.load_config
                def neutral_config(cli):
                    config = original_load(cli)
                    config.target_avg = 1.
                    config.age_decay_floor = 1.
                    config.age_relative = True
                    record["routing_parameter_overrides"] = {"target_avg": 1., "age_decay_floor": 1., "age_relative": True}
                    record["weights_or_routing_modified"] = True
                    record["model_weights_modified"] = False
                    return config
                official.load_config = neutral_config
                import tethermem.routing as routing
                original_split = routing._split_query_sdpa
                def check_neutral(*positional, **keywords):
                    # Accuracy-only control: include explicit guard overhead,
                    # and never present its latency as an optimized backend.
                    for index, key in ((4, "bias_subj"), (5, "bias_bg")):
                        bias = positional[index] if len(positional) > index else keywords.get(key)
                        if bias is not None and bool(torch.count_nonzero(bias)):
                            raise ValueError("neutral control unexpectedly has nonzero bias")
                    record["neutral_guard_calls"] = record.get("neutral_guard_calls", 0) + 1
                    return original_split(*positional, **keywords)
                routing._split_query_sdpa = check_neutral
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
