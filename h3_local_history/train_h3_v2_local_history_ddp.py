#!/usr/bin/env python3
"""Variable-window H3 v2 local-history training with limited attention LoRA."""
from __future__ import annotations

import argparse
import datetime
from contextlib import nullcontext
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.autograd.graph import save_on_cpu
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy


def bootstrap() -> None:
    execution = Path(__file__).resolve().parent
    offline = execution / "offline_h3_python_20261013"
    for path in (offline, execution / "overlay", execution, execution / "vdn-minimax-h3", execution / "vdn-minimax-h3" / "diffusers" / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))



class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        # Keep FP32 master weights for the optimizer, but do the long-sequence
        # LoRA matmuls in the frozen projection's compute dtype.  The previous
        # path promoted the [B, S, out_features] update to FP32 before casting
        # it back to BF16; for H3's packed sequence that transient is several
        # GiB per projection and is pure activation overhead.
        compute_dtype = base.dtype
        lora_input = value.to(compute_dtype)
        lora_a = self.lora_a.to(compute_dtype)
        lora_b = self.lora_b.to(compute_dtype)
        update = F.linear(F.linear(lora_input, lora_a), lora_b) * self.scale
        return base + update


def install_lora(transformer: nn.Module, rank: int, alpha: float) -> nn.ModuleDict:
    modules = nn.ModuleDict()
    blocks = transformer.transformer_blocks[-4:]
    for offset, block in enumerate(blocks):
        group = nn.ModuleDict()
        group["to_q"] = LoRALinear(block.attn.to_q, rank, alpha)
        group["to_k"] = LoRALinear(block.attn.to_k, rank, alpha)
        group["to_v"] = LoRALinear(block.attn.to_v, rank, alpha)
        group["to_out"] = LoRALinear(block.attn.to_out[0], rank, alpha)
        block.attn.to_q = group["to_q"]
        block.attn.to_k = group["to_k"]
        block.attn.to_v = group["to_v"]
        block.attn.to_out[0] = group["to_out"]
        modules[str(len(transformer.transformer_blocks) - 4 + offset)] = group
    return modules


def legal_history_start(target_start: int, source_fps: float, h_max_sec: float) -> int:
    target_pixel = 0 if target_start == 0 else 5 + 17 * ((target_start - 2) // 5)
    desired = max(0, target_pixel - round(h_max_sec * source_fps))
    if desired < 5:
        return 0
    # Round the start forward to the next legal VAE boundary, so the retained
    # interval never exceeds H_max (rounding backward can add 17 source frames).
    return 2 + 5 * math.ceil((desired - 5) / 17)


def main() -> None:
    bootstrap()
    from src.config.inference import InferenceConfig
    from src.inference.utils.assemble import build_inference_model
    from h3_spatial_history_flow import SpatialHistoryProjection, install_history_attention, matched_spatial_batch, velocity_loss

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--base-source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--h-max-sec", type=float, default=8.0)
    parser.add_argument(
        "--checkpoint-segments",
        type=int,
        default=8,
        help="number of transformer block groups for activation checkpointing; 1 keeps the legacy per-block path",
    )
    parser.add_argument(
        "--attention-query-chunk-size",
        type=int,
        default=65536,
        help="maximum packed-sequence query rows materialized per history-attention call",
    )
    parser.add_argument(
        "--ff-chunk-size",
        type=int,
        default=16384,
        help="packed-sequence rows per SwiGLU FFN chunk; 0 keeps the native full-sequence FFN",
    )
    parser.add_argument(
        "--activation-save-on-cpu",
        action="store_true",
        help="offload tensors saved for backward to host RAM; useful for full original windows",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=0,
        help="number of uniformly sampled history latent frames; 0 keeps every frame in the bounded window",
    )
    parser.add_argument("--history-frames-long", type=int, default=-1)
    parser.add_argument("--history-long-threshold", type=int, default=117)
    parser.add_argument(
        "--block-fsdp",
        action="store_true",
        help="wrap each H3 transformer block in FULL_SHARD FSDP to shard frozen trunk and adapter parameters",
    )
    parser.add_argument(
        "--full-fsdp",
        action="store_true",
        help="wrap the complete H3 transformer in FULL_SHARD FSDP (one model all-gather per forward)",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="train only the spatial history projection; keep the frozen H3 trunk unchanged",
    )
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.index).read_text().splitlines() if line.strip()]
    if not rows or any(row.get("training_ready") is not True for row in rows):
        raise ValueError("training index is not admitted")
    index_sha = hashlib.sha256(Path(args.index).read_bytes()).hexdigest()
    # FSDP all-gathers are synchronized at every wrapped block.  Original
    # annotation windows have intentionally variable target lengths, so a
    # short rank may wait several minutes for the longest rank's attention
    # recompute.  The default 10-minute NCCL group timeout is too aggressive
    # for this full-window smoke/training contract.
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    try:
        def stage(name: str) -> None:
            current_device = torch.cuda.current_device() if torch.cuda.is_available() else None
            print(json.dumps({
                "rank": rank,
                "stage": name,
                "time": time.time(),
                "allocated_gib": torch.cuda.memory_allocated(current_device) / 2**30 if current_device is not None else 0.0,
                "reserved_gib": torch.cuda.memory_reserved(current_device) / 2**30 if current_device is not None else 0.0,
            }), flush=True)

        stage("dist_initialized")
        if len(rows) < world:
            raise ValueError("training rows fewer than ranks")
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        torch.manual_seed(args.seed)
        cfg = InferenceConfig()
        cfg.base_source = args.base_source
        cfg.vae_source = args.base_source
        cfg.kernels.inference_kernels = False
        cfg.kernels.softmax_backend = "ref"
        stage("model_build_start")
        model = build_inference_model(cfg, str(device), load_decoders=False, log=rank == 0)
        stage("model_build_done")
        model.transformer.requires_grad_(False)
        model.transformer.set_attention_backend("native")
        model.transformer.gradient_checkpointing = True
        model.transformer.gradient_checkpointing_segments = max(1, args.checkpoint_segments)
        for block in model.transformer.transformer_blocks:
            block.ff_chunk_size = max(0, args.ff_chunk_size)
        model.transformer._gradient_checkpointing_func = lambda module, *inputs: checkpoint(module, *inputs, use_reentrant=False)
        stage("attention_install_start")
        processors = install_history_attention(
            model.transformer,
            query_chunk_size=args.attention_query_chunk_size,
            query_checkpoint=not (args.block_fsdp or args.full_fsdp),
        )
        stage("attention_install_done")
        projection = SpatialHistoryProjection(model.transformer.proj_in).to(device)
        if args.no_lora:
            lora = nn.ModuleDict()
            stage("lora_disabled_projection_only")
        else:
            stage("lora_install_start")
            lora = install_lora(model.transformer, args.lora_rank, args.lora_alpha).to(device)
            stage("lora_install_done")
        if args.full_fsdp:
            ignored_states = [
                parameter for parameter in model.transformer.parameters() if parameter.dtype != torch.bfloat16
            ]
            model.transformer = FSDP(
                model.transformer,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                use_orig_params=True,
                device_id=device,
                sync_module_states=False,
                ignored_states=ignored_states,
            )
            stage("full_fsdp_install_done")
        elif args.block_fsdp:
            # Shard one DiT block at a time.  Wrapping the whole transformer
            # would all-gather the complete 50-block trunk and lose the memory
            # benefit; block granularity keeps only the active block's full
            # parameters resident during its forward/recompute.
            for index, block in enumerate(model.transformer.transformer_blocks):
                # H3 keeps a few block parameters and the FP32 LoRA masters in
                # float32. FSDP flattening requires a uniform dtype, so leave
                # those small tensors replicated and shard the dominant BF16
                # trunk.
                ignored_states = [
                    parameter for parameter in block.parameters() if parameter.dtype != torch.bfloat16
                ]
                model.transformer.transformer_blocks[index] = FSDP(
                    block,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    use_orig_params=True,
                    device_id=device,
                    sync_module_states=False,
                    ignored_states=ignored_states,
                )
            stage("block_fsdp_install_done")
        # With block-FSDP the LoRA modules are already owned by their wrapped
        # transformer blocks. Registering the same modules again in a separate
        # ModuleDict makes FSDP see aliased parameter ownership and can force a
        # full-trunk all-gather. Keep only the projection in the side module;
        # collect LoRA parameters directly from the wrapped blocks.
        if args.full_fsdp or args.block_fsdp:
            trainables = nn.ModuleDict({"projection": projection}).to(device)
            optimizer_params = list(trainables.parameters()) + [
                parameter for parameter in model.transformer.parameters() if parameter.requires_grad
            ]
        elif args.no_lora:
            trainables = nn.ModuleDict({"projection": projection}).to(device)
            optimizer_params = list(trainables.parameters())
        else:
            trainables = nn.ModuleDict({"projection": projection, "lora": lora}).to(device)
            optimizer_params = list(trainables.parameters())
        for parameter in trainables.parameters():
            dist.broadcast(parameter.data, src=0)
        optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=1e-4)
        noise = torch.Generator(device=device).manual_seed(args.seed + 1000 + rank)
        output = Path(args.out)
        output.mkdir(parents=True, exist_ok=True)
        for step in range(1, args.steps + 1):
            start = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            row = rows[((step - 1) * world + rank) % len(rows)]
            stage(f"step_{step}_sample_load_start")
            video = torch.load(row["video_path"], map_location="cpu", weights_only=True)
            audio = torch.load(row["audio_path"], map_location="cpu", weights_only=True)
            mask = torch.load(row["audio_valid_mask_path"], map_location="cpu", weights_only=True)
            prompt = torch.load(row["prompt_path"], map_location="cpu", weights_only=True)
            stage(f"step_{step}_sample_load_done")
            target_start = int(row["video_latent_start"])
            target_end = int(row["video_latent_end"])
            history_start = legal_history_start(target_start, 24.0, args.h_max_sec)
            history_frames = args.history_frames
            if args.history_frames_long >= 0 and (target_end - target_start) >= args.history_long_threshold:
                history_frames = args.history_frames_long
            batch = matched_spatial_batch(
                video, audio, prompt["prompt_embeds"], prompt["text_token_tags"], mask, noise, device,
                history_start_latents=history_start, history_end_latents=target_start,
                target_start_latents=target_start, target_end_latents=target_end,
                history_frames=history_frames, target_latents=target_end - target_start,
            )
            stage(f"step_{step}_batch_done")
            for processor in processors:
                processor.prefix_rows = batch["prefix_rows"]
            optimizer.zero_grad(set_to_none=True)
            batch["inputs"]["state_hidden_states"] = projection(batch["history_rows"])
            activation_context = save_on_cpu(pin_memory=False) if args.activation_save_on_cpu else nullcontext()
            with activation_context:
                pred_v, pred_a = model.transformer(**batch["inputs"])
                loss, video_loss, audio_loss = velocity_loss(pred_v, pred_a, batch)
            stage(f"step_{step}_forward_done")
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            stage(f"step_{step}_backward_done")
            manual_params = list(trainables.parameters())
            if not (args.full_fsdp or args.block_fsdp):
                manual_params = optimizer_params
            gradients = [parameter for parameter in manual_params if parameter.grad is not None]
            for parameter in gradients:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(world)
            grad_norm = torch.nn.utils.clip_grad_norm_(optimizer_params, 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("nonfinite gradient")
            optimizer.step()
            torch.cuda.synchronize(device)
            record = {
                "step": step, "rank": rank, "world_size": world, "sample_id": row["sample_id"],
                "loss": float(loss), "video_loss": float(video_loss), "audio_loss": float(audio_loss),
                "grad_norm": float(grad_norm), "history_start_latent": history_start,
                "target_start_latent": target_start, "target_end_latent": target_end,
                "step_seconds": time.monotonic() - start,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "trainable_parameters": sum(parameter.numel() for parameter in optimizer_params),
                "checkpoint_segments": args.checkpoint_segments,
                "attention_query_chunk_size": args.attention_query_chunk_size,
                "block_fsdp": args.block_fsdp,
                "training_kind": "variable_window_teacher_forced_projection_plus_limited_attention_lora",
                "data_version": "h3_temporal_aligned_v2",
            }
            with (output / f"rank{rank}.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if rank == 0:
                print(json.dumps(record), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                if rank == 0:
                    payload = {
                        "step": step, "world_size": world, "index_sha256": index_sha,
                        "trainables": trainables.state_dict(), "lora": lora.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "args": vars(args), "training_kind": "variable_window_teacher_forced_projection_plus_lora",
                        "event_eval_ready": False, "generated_history_training": False,
                    }
                    temporary = output / "checkpoint.pt.tmp"
                    torch.save(payload, temporary)
                    temporary.replace(output / "checkpoint.pt")
                dist.barrier()
        if rank == 0:
            (output / "TRAINING_PASS.json").write_text(json.dumps({
                "status": "h3_v2_variable_window_training_pass",
                "steps": args.steps, "world_size": world, "data_version": "h3_temporal_aligned_v2",
                "h_max_sec": args.h_max_sec, "event_eval_ready": False,
            }, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
