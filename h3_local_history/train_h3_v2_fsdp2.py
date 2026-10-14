#!/usr/bin/env python3
"""Original-window matched-flow training with mixed-dtype, block-wise FSDP2.

Frozen native H3 + FP32 history projection + limited module-level attention LoRA.
FlashAttention query tiles and FFN tiles change activation lifetime, not context.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, offload_wrapper, CheckpointImpl,
)

from train_h3_v2_local_history_ddp import bootstrap, install_lora, legal_history_start


class TrainingModel(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        from h3_spatial_history_flow import SpatialHistoryProjection
        self.transformer = transformer
        self.projection = SpatialHistoryProjection(transformer.proj_in)

    def forward(self, batch):
        from h3_spatial_history_flow import velocity_loss
        inputs = dict(batch['inputs'])
        inputs['state_hidden_states'] = self.projection(batch['history_rows'])
        video, audio = self.transformer(**inputs)
        return velocity_loss(video, audio, batch)


def shard_model(model, mesh, offload=True):
    # Preserve the native mixed precision and FP32 LoRA master parameters.
    # FSDP2 groups parameters by dtype rather than flattening BF16 and FP32 together.
    policy = MixedPrecisionPolicy(cast_forward_inputs=False)
    model.transformer.gradient_checkpointing = False
    replicated_masters = set()
    for i, block in enumerate(model.transformer.transformer_blocks):
        ignored = {p for p in block.parameters() if p.dtype != torch.bfloat16}
        replicated_masters.update(ignored)
        wrapped = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        if offload:
            # Only block checkpoint inputs go to host RAM. Fused attention's
            # recompute/backward stays on CUDA; do not offload every softmax tile.
            wrapped = offload_wrapper(wrapped)
        model.transformer.transformer_blocks[i] = wrapped
        fully_shard(wrapped, mesh=mesh, mp_policy=policy, reshard_after_forward=True, ignored_params=ignored)
    # PyTorch 2.7 requires one original dtype per FSDP group. Give the native
    # FP32 heads and history projection their own groups; tiny FP32 LoRA
    # masters stay replicated and explicitly all-reduce gradients.
    for name in ('proj_in','audio_proj_in','time_embedder','proj_out','audio_proj_out'):
        fully_shard(getattr(model.transformer,name),mesh=mesh,mp_policy=policy)
    fully_shard(model.projection,mesh=mesh,mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True, ignored_params=replicated_masters)
    return model


def local_tensor(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def global_clip(parameters, device):
    norm2 = torch.zeros((), dtype=torch.float64, device=device)
    for p in parameters:
        if p.grad is not None:
            term = local_tensor(p.grad).double().square().sum()
            norm2 += term if isinstance(p,DTensor) else term/dist.get_world_size()
    dist.all_reduce(norm2)
    norm = norm2.sqrt()
    if not torch.isfinite(norm):
        raise RuntimeError('nonfinite global gradient norm')
    coefficient = (1.0 / (norm + 1e-6)).clamp(max=1.0)
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(coefficient.to(p.grad.dtype))
    return float(norm)


def cpu_local(value):
    if isinstance(value, torch.Tensor):
        return local_tensor(value).detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_local(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_local(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_local(v) for v in value)
    return value


def save_checkpoint(model, optimizer, generator, out, step, args, index_sha, rank, world):
    # All ranks participate in adapter gathering; frozen 30B trunk is never copied/saved.
    full = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            tensor = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
            if rank == 0:
                full[name] = tensor.detach().cpu()
    rank_state = {
        'optimizer': cpu_local(optimizer.state_dict()),
        'noise_rng': generator.get_state().cpu(),
        'cpu_rng': torch.get_rng_state().cpu(),
        'cuda_rng': torch.cuda.get_rng_state().cpu(),
    }
    target = out / f'optimizer_rank{rank}.pt'
    temporary = target.with_suffix('.pt.tmp')
    torch.save(rank_state, temporary)
    temporary.replace(target)
    dist.barrier()
    if rank == 0:
        payload = {
            'schema': 'h3_local_history_fsdp2_adapters_v1',
            'step': step, 'world_size': world, 'index_sha256': index_sha,
            'adapters': full, 'args': vars(args),
            'training_kind': 'teacher_forced_matched_velocity_projection_plus_attention_lora',
            'history_policy': 'complete_contiguous_recent_window',
            'event_eval_ready': False, 'generated_history_training': False,
        }
        temporary = out / 'checkpoint.pt.tmp'
        torch.save(payload, temporary)
        temporary.replace(out / 'checkpoint.pt')
    dist.barrier()


def resume_checkpoint(model, optimizer, generator, path, mesh, rank, device, world, index_sha):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt['world_size'] != world or ckpt['index_sha256'] != index_sha:
        raise ValueError('resume requires identical world size and training index')
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in ckpt['adapters'].items():
            parameter = parameters[name]
            restored = distribute_tensor(value.to(device), mesh, parameter.placements) if isinstance(parameter,DTensor) else value.to(device)
            parameter.copy_(restored)
    state = torch.load(Path(path).parent / f'optimizer_rank{rank}.pt', map_location='cpu', weights_only=False)
    saved = state['optimizer']
    # Reconstruct optimizer DTensors on the same shard placements.
    for group, saved_group in zip(optimizer.param_groups, saved['param_groups']):
        for parameter, ident in zip(group['params'], saved_group['params']):
            parameter_state = saved['state'].get(ident, {})
            for key, value in list(parameter_state.items()):
                if isinstance(value, torch.Tensor) and value.ndim and key in ('exp_avg', 'exp_avg_sq'):
                    parameter_state[key] = DTensor.from_local(
                        value.to(device), mesh, parameter.placements,
                        shape=parameter.shape, stride=parameter.stride(), run_check=False,
                    ) if isinstance(parameter,DTensor) else value.to(device)
    optimizer.load_state_dict(saved)
    generator.set_state(state['noise_rng'].cpu().to(torch.uint8))
    torch.set_rng_state(state['cpu_rng'].cpu().to(torch.uint8))
    torch.cuda.set_rng_state(state['cuda_rng'].cpu().to(torch.uint8), device)
    return int(ckpt['step'])


def main():
    bootstrap()
    from src.config.inference import InferenceConfig
    from src.inference.utils.assemble import build_inference_model
    from h3_spatial_history_flow import install_history_attention, matched_spatial_batch
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True)
    ap.add_argument('--base-source', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--save-every', type=int, default=10)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--h-max-sec', type=float, default=8.0)
    ap.add_argument('--lora-rank', type=int, default=8)
    ap.add_argument('--lora-alpha', type=float, default=8.0)
    ap.add_argument('--query-tile', type=int, default=8192)
    ap.add_argument('--ff-tile', type=int, default=8192)
    ap.add_argument('--no-activation-offload', action='store_true')
    ap.add_argument('--resume')
    ap.add_argument('--seed', type=int, default=20261013)
    args = ap.parse_args()
    raw = Path(args.index).read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows or any(r.get('training_ready') is not True for r in rows):
        raise ValueError('index is not training admitted')
    index_sha = hashlib.sha256(raw).hexdigest()
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def stage(name, **extra):
        record = dict(rank=rank, stage=name, time=time.time(),
            allocated_gib=torch.cuda.memory_allocated(device)/2**30,
            peak_gib=torch.cuda.max_memory_allocated(device)/2**30)
        record.update(extra)
        print(json.dumps(record), flush=True)

    try:
        torch.manual_seed(args.seed)
        torch.set_num_threads(2)
        cfg = InferenceConfig()
        cfg.base_source = cfg.vae_source = args.base_source
        cfg.kernels.inference_kernels = False
        stage('model_load_start')
        built = build_inference_model(cfg, str(device), load_decoders=False, log=(rank == 0))
        transformer = built.transformer
        transformer.requires_grad_(False)
        processors = install_history_attention(transformer, args.query_tile, query_checkpoint=True)
        install_lora(transformer, args.lora_rank, args.lora_alpha)
        for block in transformer.transformer_blocks:
            block.ff_chunk_size = args.ff_tile
        model = TrainingModel(transformer).to(device)
        for parameter in model.parameters():
            if parameter.requires_grad:
                dist.broadcast(parameter.detach(), src=0)
        stage('model_loaded', parameter_count=sum(p.numel() for p in model.parameters()))
        mesh = init_device_mesh('cuda', (world,))
        model = shard_model(model, mesh, offload=not args.no_activation_offload)
        torch.cuda.empty_cache()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4, foreach=False)
        noise = torch.Generator(device=device).manual_seed(args.seed + 1000 + rank)
        stage('fsdp2_ready', local_parameter_numel=sum(local_tensor(p).numel() for p in model.parameters()),
              trainable_global_numel=sum(p.numel() for p in parameters))
        first_step = resume_checkpoint(model, optimizer, noise, args.resume, mesh, rank, device, world, index_sha) if args.resume else 0
        for step in range(first_step + 1, args.steps + 1):
            row = rows[((step-1)*world + rank) % len(rows)]
            torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            video, audio, mask, prompt = (torch.load(row[k], map_location='cpu', weights_only=True)
                for k in ('video_path', 'audio_path', 'audio_valid_mask_path', 'prompt_path'))
            target_start, target_end = int(row['video_latent_start']), int(row['video_latent_end'])
            history_start = legal_history_start(target_start, 24.0, args.h_max_sec)
            batch = matched_spatial_batch(video, audio, prompt['prompt_embeds'], prompt['text_token_tags'],
                mask, noise, device, history_frames=0,
                history_start_latents=history_start, history_end_latents=target_start,
                target_start_latents=target_start, target_end_latents=target_end,
                target_audio_start_latents=int(row['audio_latent_start']),
                target_audio_end_latents=int(row['audio_latent_end']))
            for processor in processors:
                processor.prefix_rows = batch['prefix_rows']
            optimizer.zero_grad(set_to_none=True)
            before = [local_tensor(p).detach().clone() for p in parameters]
            stage('forward_start', step=step, window_id=row['window_id'],
                history_latents=target_start-history_start, target_latents=target_end-target_start,
                packed_tokens=len(batch['inputs']['position_ids']))
            loss, lv, la = model(batch)
            stage('forward_done', step=step, loss=float(loss))
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss')
            loss.backward()
            stage('backward_done', step=step)
            for parameter in parameters:
                if not isinstance(parameter,DTensor):
                    if parameter.grad is None:
                        parameter.grad=torch.zeros_like(parameter)
                    dist.all_reduce(parameter.grad)
                    parameter.grad.div_(world)
            grad_norm = global_clip(parameters, device)
            optimizer.step()
            delta = sum((local_tensor(p).detach()-b).double().square().sum() for p,b in zip(parameters,before))
            dist.all_reduce(delta)
            if not torch.isfinite(delta) or delta <= 0:
                raise RuntimeError('optimizer did not update trainable parameters')
            torch.cuda.synchronize()
            record = dict(step=step, rank=rank, world_size=world, window_id=row['window_id'],
                sample_id=row['sample_id'], loss=float(loss), video_loss=float(lv), audio_loss=float(la),
                global_grad_norm=grad_norm, global_update_l2=float(delta.sqrt()),
                step_seconds=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                history_start_latent=history_start, history_end_latent=target_start,
                target_start_latent=target_start, target_end_latent=target_end,
                history_frame_ids=batch['contract']['history_frame_ids'],
                packed_tokens=len(batch['inputs']['position_ids']), framework='fsdp2_full_shard',
                attention='flash_sdpa_prefix_block', trainables='projection_and_module_attention_lora',
                index_sha256=index_sha, event_eval_ready=False)
            with (out/f'rank{rank}.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            stage('optimizer_done', **record)
            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(model, optimizer, noise, out, step, args, index_sha, rank, world)
                stage('checkpoint_saved', step=step)
            del loss, lv, la, batch, before, video, audio, mask, prompt
        dist.barrier()
        if rank == 0:
            (out/'TRAINING_PASS.json').write_text(json.dumps(dict(
                status='h3_fsdp2_real_training_pass', steps=args.steps, world_size=world,
                index_sha256=index_sha, original_windows=True, full_recent_history=True,
                trainables='projection_plus_attention_lora', event_eval_ready=False),indent=2)+'\n')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
