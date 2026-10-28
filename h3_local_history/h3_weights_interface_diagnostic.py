"""Isolate weight updates, text visibility and text time without any history.

All eight ranks load slices of the SAME checkpoint. Four rule variants and two
seeds share one FSDP session; different checkpoints require different sessions.
This is an explicitly changed-interface diagnostic, never a streaming score.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from train_h3_v2_local_history_ddp import bootstrap
bootstrap()
from h3_shared_loader import load_session
from h3_shared_stage1 import shared_modules
from src.inference.render import generate_latents, decode_and_save


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


# Stage1B's explicit sink+recent causal path. This lives in the already
# published diagnostic worker so the short checkpoint gate uses the same
# public commit as the existing A200 diagnosis; no legacy fixed-KV processor
# is silently reused.
@dataclass(frozen=True)
class Stage1BLayout:
    block_ids: torch.Tensor
    video_frames: torch.Tensor
    local_frames: int = 12
    sink_frames: int = 3
    current_block: int = 0

    def groups(self):
        return [int(x) for x in torch.unique(self.block_ids, sorted=True).tolist()]

    def visible_indices(self, block):
        text = self.block_ids == -1
        if block == -1:
            current = self.block_ids == self.current_block
            return torch.where(text | current)[0]
        current = self.block_ids == block
        frames = self.video_frames[current & (self.video_frames >= 0)]
        if not frames.numel():
            raise ValueError('Stage1B current block has no video rows')
        end = int(frames.max()) + 1
        budget = self.local_frames - self.sink_frames
        if int(frames.max() - frames.min() + 1) > budget:
            raise ValueError('Stage1B current block exceeds sink-reserved window')
        past = (self.block_ids >= 0) & (self.block_ids < block) & (self.video_frames >= 0)
        keep = (self.video_frames < self.sink_frames) | (self.video_frames >= max(self.sink_frames, end - budget))
        return torch.cat((torch.where(text)[0], torch.where(past & keep)[0], torch.where(current)[0]))


@dataclass(frozen=True)
class Stage1BContext:
    layout: Stage1BLayout
    cached: object = None
    write_bank: object = None
    modulation_segments: tuple = None
    text_reads_history: bool = False


class Stage1BCausalAttention:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, layer_id, backend='torch_flash'):
        self.layer_id, self.backend = layer_id, backend
        self.query_checkpoint = False

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None, is_causal=False):
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
        from h3_attention_backend import flash_attention
        if is_causal or not isinstance(attention_mask, Stage1BContext):
            raise ValueError('Stage1B requires explicit causal context')
        context, layout = attention_mask, attention_mask.layout
        if hidden_states.shape[1] != layout.block_ids.numel():
            raise ValueError('Stage1B packed rows do not match causal layout')
        q = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1)))
        k = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1)))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1)).transpose(1, 2)
        if rotary_emb is not None:
            q = _apply_rotary_emb(q, *rotary_emb)
            k = _apply_rotary_emb(k, *rotary_emb)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        output = torch.zeros_like(q)
        past = context.cached.get(self.layer_id) if context.cached is not None else None
        if context.cached is not None and len([g for g in layout.groups() if g >= 0]) != 1:
            raise ValueError('Stage1B cache reads require one current block')
        for group in layout.groups():
            query_rows = torch.where(layout.block_ids == group)[0]
            visible = layout.visible_indices(group)
            keys, values = k.index_select(2, visible), v.index_select(2, visible)
            if (group >= 0 or context.text_reads_history) and past is not None:
                frames = layout.video_frames[(layout.block_ids == layout.current_block) & (layout.video_frames >= 0)]
                start, end = int(frames.min()), int(frames.max()) + 1
                if (past.frames >= start).any():
                    raise ValueError('Stage1B cache contains current or future frames')
                budget = layout.local_frames - layout.sink_frames
                keep = (past.frames < layout.sink_frames) | (past.frames >= max(layout.sink_frames, end - budget))
                text = layout.block_ids[visible] == -1
                keys = torch.cat((keys[:, :, text], past.key[:, :, keep], keys[:, :, ~text]), dim=2)
                values = torch.cat((values[:, :, text], past.value[:, :, keep], values[:, :, ~text]), dim=2)
            attended = flash_attention(q.index_select(2, query_rows), keys, values, self.backend)
            output = output.index_copy(2, query_rows, attended)
        if context.write_bank is not None:
            mask = layout.video_frames >= 0
            context.write_bank.commit(self.layer_id, k[:, :, mask], v[:, :, mask], layout.video_frames[mask])
        return attn.to_out[1](attn.to_out[0](output.transpose(1, 2).flatten(2, 3)))


def stage1b_segments(indices):
    values, counts = torch.unique_consecutive(indices, return_counts=True)
    values, counts = values.cpu().tolist(), counts.cpu().tolist()
    start, result = 0, []
    for value, count in zip(values, counts):
        result.append((start, start + count, value)); start += count
    return tuple(result)


def install_stage1b_processors(model):
    processors = []
    for layer, wrapped in enumerate(model.transformer.transformer_blocks):
        block = wrapped
        while hasattr(block, '_checkpoint_wrapped_module'):
            block = block._checkpoint_wrapped_module
        processor = Stage1BCausalAttention(layer)
        block.attn.set_processor(processor)
        block.ff_chunk_size, block.ff_chunk_checkpoint = 8192, False
        processors.append(processor)
    return processors


def stage1b_context(batch, chunk, start, end, bank, write_bank=None, text_reads_history=False):
    x = batch['inputs']; seq = len(x['position_ids'])
    rows_per_frame = 48 // 2 * (84 // 2)
    blocks = torch.full((seq,), chunk, device=x['position_ids'].device, dtype=torch.long)
    blocks[x['text_indices']] = -1
    frames = torch.full_like(blocks, -1)
    frames[x['video_indices']] = torch.arange(start, end, device=frames.device).repeat_interleave(rows_per_frame)
    return Stage1BContext(Stage1BLayout(blocks, frames, 12, 3, chunk),
                          cached=bank.snapshot() if bank is not None else None,
                          write_bank=write_bank,
                          modulation_segments=stage1b_segments(x['timestep_indices'] * 3 + x['token_tags']),
                          text_reads_history=text_reads_history)


def stage1b_clock(batch, tv, ta):
    x = batch['inputs']; row_t = torch.full((len(x['position_ids']),), float(tv), device=x['position_ids'].device)
    row_t[x['state_indices']] = 1.; row_t[x['audio_indices']] = float(ta)
    x['timestep'], x['timestep_indices'] = torch.unique(row_t, sorted=True, return_inverse=True)


def stage1b_chunk(video, audio, prompt, tags, generator, device, start, end, audio_end):
    from h3_spatial_history_flow import matched_spatial_batch
    batch = matched_spatial_batch(video, audio, prompt, tags,
        torch.ones(audio.shape[-1], dtype=torch.bool), generator, device,
        history_frames=0, history_start_latents=0, history_end_latents=0,
        target_start_latents=start, target_end_latents=end,
        target_audio_start_latents=0 if start == 0 else round((5 + 17 * ((start - 2) // 5)) * 40 / 24),
        target_audio_end_latents=audio_end)
    batch.pop('target_video', None); batch.pop('target_audio', None)
    batch['prefix_rows'] = batch['inputs']['text_indices'].numel()
    batch['inputs']['state_hidden_states'] = batch['inputs']['encoder_hidden_states'].new_empty((1, 0, 5376), dtype=torch.bfloat16)
    return batch


def stage1b_clocks(u):
    return 1 - 12 * u / (1 + 11 * u), 1 - 3 * u / (1 + 2 * u)


def stage1b_short(request_path, request):
    if not (request.get('diagnostic_only') is True and request.get('mode') == 'stage1b_current_only_short'
            and request.get('attention_contract') == 'STAGE1B_CURRENT_ONLY_V1'
            and request.get('NFE') == 4 and request.get('chunks') == 4):
        raise ValueError('Use the explicit Stage1B current-only four-step short contract')
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local); torch.set_num_threads(2); device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=40), device_id=device)
    try:
        if dist.get_world_size() != 8: raise ValueError('Stage1B short gate requires eight ranks')
        identities = [None] * 8; dist.all_gather_object(identities, file_sha(request_path))
        if len(set(identities)) != 1: raise ValueError('Mixed Stage1B short requests')
        error = [None]
        if rank == 0:
            try:
                if file_sha(request['weights_spec']['checkpoint']) != request['weights_spec']['checkpoint_sha256']:
                    error[0] = 'Stage1B checkpoint hash differs from frozen request'
            except Exception as exc: error[0] = repr(exc)
        dist.broadcast_object_list(error, src=0)
        if error[0] is not None: raise ValueError(error[0])
        prompt = torch.load(request['prompt_path'], map_location='cpu', weights_only=True)
        if prompt['prompt'] != request['prompt'] or file_sha(request['prompt_path']) != request['prompt_file_sha256']:
            raise ValueError('Stage1B prompt encoding changed')
        if tensor_sha(prompt['prompt_embeds']) != request['prompt_tensor_sha256']:
            raise ValueError('Stage1B prompt tensor changed')
        session = load_session(request['weights_spec'], device, dense=True, stage1b_current_only=True)
        install_stage1b_processors(session['model'])
        # FSDP2 shards the context embedder output dimension. Replicate the
        # identical prompt rows as a DTensor before the first Linear; passing a
        # plain Tensor to a DTensor weight fails before any DiT attention runs.
        from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
        context_weight = session['model'].transformer.context_embedder.weight
        context_mesh = context_weight.device_mesh if isinstance(context_weight, DTensor) else None
        generator = torch.Generator(device=device).manual_seed(int(request['seed']))
        bank = None
        from h3_stage1_clock_probe import tensor_sha as clock_tensor_sha
        from h3_worldattention_causal import VideoKVBank
        bank = VideoKVBank(local_frames=9, sink_frames=3)
        video_template = torch.zeros(24, 22, 48, 84, dtype=torch.bfloat16)
        audio_template = torch.zeros(2, 32, 122, dtype=torch.bfloat16)
        videos, audios, records = [], [], []; started = time.monotonic()
        with torch.inference_mode():
            for ci, (start, end) in enumerate(((0, 7), (7, 12), (12, 17), (17, 22))):
                audio_end = 122 if end == 22 else round((5 + 17 * ((end - 2) // 5)) * 40 / 24)
                batch = stage1b_chunk(video_template, audio_template, prompt['prompt_embeds'], prompt['text_token_tags'], generator, device, start, end, audio_end)
                x = batch['inputs']
                if context_mesh is not None and not isinstance(x['encoder_hidden_states'], DTensor):
                    x['encoder_hidden_states'] = distribute_tensor(x['encoder_hidden_states'], context_mesh, [Replicate()])
                video = torch.randn(x['hidden_states'].shape, device=device, generator=generator); audio = torch.randn(x['audio_hidden_states'].shape, device=device, generator=generator)
                first_noise = (clock_tensor_sha(video), clock_tensor_sha(audio)); step_times=[]
                for i in range(4):
                    tv, ta = stage1b_clocks(1 - i / 4); x['hidden_states'], x['audio_hidden_states'] = video, audio; stage1b_clock(batch, tv, ta)
                    tick = time.monotonic(); pv, pa = session['model'].transformer(**x, attention_mask=stage1b_context(batch, ci, start, end, bank)); torch.cuda.synchronize(device); step_times.append(time.monotonic() - tick)
                    clean_v = (video.float() + (1 - tv) * pv.float()).to(torch.bfloat16); clean_a = (audio.float() + (1 - ta) * pa.float()).to(torch.bfloat16)
                    if i < 3:
                        nv, na = stage1b_clocks(1 - (i + 1) / 4)
                        video = (nv * clean_v.float() + (1 - nv) * torch.randn(clean_v.shape, device=device, generator=generator)).to(torch.bfloat16)
                        audio = (na * clean_a.float() + (1 - na) * torch.randn(clean_a.shape, device=device, generator=generator)).to(torch.bfloat16)
                    else: video, audio = clean_v, clean_a
                stage1b_clock(batch, 1., 1.); x['hidden_states'], x['audio_hidden_states'] = video.detach(), audio.detach()
                session['model'].transformer(**x, attention_mask=stage1b_context(batch, ci, start, end, bank, bank))
                if not bool(torch.isfinite(video).all() and torch.isfinite(audio).all()): raise ValueError('Stage1B short output is nonfinite')
                videos.append(video.cpu()); audios.append(audio.cpu())
                records.append(dict(chunk=ci, start_latent=start, end_latent=end, finite=True, NFE=4, clean_writes=1, old_history_queries_recomputed=0, KV_bytes=bank.bytes, initial_video_sha256=first_noise[0], initial_audio_sha256=first_noise[1], per_step_seconds=step_times))
        if rank == 0:
            from sample_demo import unpack_video
            decoded_video = unpack_video(torch.cat(videos, dim=1)[0], 22, 48, 84).to(device)
            decoded_audio = torch.cat([v[0].reshape(2, -1, 32) for v in audios], dim=1).permute(0, 2, 1).to(device)
            out = Path(os.environ['INFER_OUTPUT_DIR']); out.mkdir(parents=True, exist_ok=True); path = out / 'STAGE1B_STEP25_4CHUNK.mp4'; archive = out / 'STAGE1B_STEP25_4CHUNK_LATENTS.pt'
            torch.save(dict(video=decoded_video.cpu(), audio=decoded_audio.cpu(), records=records), archive)
            session['vae'].to(device); session['audio_vae'].to(device); decode_and_save(decoded_video, decoded_audio, session['vae'], session['audio_vae'], str(path), str(device))
            result = dict(status='stage1b_step25_4chunk_video_complete_quality_pending', checkpoint_step=session['checkpoint_step'], checkpoint_sha256=request['weights_spec']['checkpoint_sha256'], contract='STAGE1B_CURRENT_ONLY_V1', frames=73, FPS=24, NFE=4, chunks=4, records=records, video=str(path), video_sha256=file_sha(path), latent_archive=str(archive), latent_archive_sha256=file_sha(archive), old_history_queries_recomputed=0, quality_verified=False, formal_benchmark_admission=False, wall_seconds=time.monotonic() - started, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, GPU=torch.cuda.get_device_name(device))
            (out / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()


def stage1b_kv_samples(bank):
    # A small value fingerprint per layer/frame, not a full KV checksum.
    return {layer: ((entry.frames * 1008 + torch.arange(entry.frames.numel(), device=entry.frames.device) % 1008),
                    torch.cat((entry.key[0, 0, :, :2], entry.value[0, 0, :, :2]), -1).clone())
            for layer, entry in bank.snapshot().items()}


def stage1b_check_samples(before, bank):
    checked = 0
    for layer, (frames, values) in before.items():
        after = bank.snapshot()[layer]
        after_ids = after.frames * 1008 + torch.arange(after.frames.numel(), device=after.frames.device) % 1008
        keep = torch.isin(frames, after_ids)
        if not keep.any(): continue
        ids = torch.searchsorted(after_ids, frames[keep])
        actual = torch.cat((after.key[0, 0, ids, :2], after.value[0, 0, ids, :2]), -1)
        if not torch.equal(values[keep], actual): raise ValueError('Previously committed video KV changed')
        checked += int(keep.sum())
    return checked


def stage1b_decode_long(session, out, latents, frames, device):
    import subprocess
    import wave
    import imageio_ffmpeg
    from h3_buffered_video_decode import BufferedVideoDecoder
    from h3_frame_writer import FrameWriter
    from sample_demo import unpack_video
    session['vae'].to(device)
    decoder = BufferedVideoDecoder(session['vae'], device)
    began = time.monotonic()
    with FrameWriter(out / 'VIDEO.mp4', 24, encoder_threads=2) as writer:
        for item in latents:
            saved = torch.load(item, map_location='cpu', weights_only=True)
            video = saved['video']
            writer.append(decoder.push(unpack_video(video[0], video.shape[1] // 1008, 48, 84).to(device)))
        writer.append(decoder.finish())
    if writer.frames != frames: raise ValueError('Decoded frame count differs from native H3 geometry')
    codec = decoder.last_summary
    session['vae'].cpu(); torch.cuda.empty_cache()
    session['audio_vae'].to(device)
    parts = [torch.load(p, map_location='cpu', weights_only=True)['audio'][0].reshape(2, -1, 32) for p in latents]
    sound = torch.cat(parts, dim=1).permute(0, 2, 1).to(device)
    avae = session['audio_vae']
    mean = torch.tensor(avae.config.latents_mean, device=device).view(1, -1, 1)
    std = torch.tensor(avae.config.latents_std, device=device).view(1, -1, 1)
    pcm = avae.decode(sound * std + mean, return_dict=False)[0].float().permute(1, 0, 2)[0]
    wav = out / 'AUDIO.wav'
    with wave.open(str(wav), 'wb') as stream:
        stream.setnchannels(2); stream.setsampwidth(2); stream.setframerate(32000)
        stream.writeframes((pcm.clamp(-1, 1).cpu().T * 32767).round().short().contiguous().numpy().tobytes())
    mux = out / 'MUX.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-nostdin', '-y',
                    '-i', str(out / 'VIDEO.mp4'), '-i', str(wav), '-c:v', 'copy', '-c:a', 'aac',
                    '-threads', '2', '-af', 'apad', '-t', str(frames / 24), str(mux)], check=True, timeout=180)
    mux.replace(out / 'VIDEO.mp4')
    session['audio_vae'].cpu(); torch.cuda.empty_cache()
    return dict(video=str(out / 'VIDEO.mp4'), video_sha256=file_sha(out / 'VIDEO.mp4'),
                audio_sha256=file_sha(wav), decoded_frames=writer.frames, codec=codec,
                seconds=time.monotonic() - began)


def stage1b_long(request_path, request):
    """Hold video-history reads fixed; change only text reading historical KV.

    Clean writer stays at the trained current-only rule in both lanes, as in
    the A200 sampling intervention. Changing sampling visibility is explicitly
    a changed-interface diagnostic, never a matching trained-contract score.
    """
    expected_policies = [dict(name='current_native', text_reads_history=False),
                         dict(name='current_history_native', text_reads_history=True)]
    if not (request.get('diagnostic_only') is True and request.get('mode') == 'stage1b_current_only_long'
            and request.get('attention_contract') == 'STAGE1B_CURRENT_ONLY_V1'
            and request.get('NFE') == 4 and request.get('chunks') == 84
            and request.get('policies') == expected_policies
            and request.get('history_recomputation_allowed') is False):
        raise ValueError('Use the frozen Stage1B 84-chunk text-history comparison')
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local); torch.set_num_threads(2); device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=40), device_id=device)
    try:
        if dist.get_world_size() != 8: raise ValueError('Stage1B long gate requires eight ranks')
        identities = [None] * 8; dist.all_gather_object(identities, file_sha(request_path))
        if len(set(identities)) != 1: raise ValueError('Mixed Stage1B long requests')
        error = [None]
        if rank == 0:
            try:
                if file_sha(request['weights_spec']['checkpoint']) != request['weights_spec']['checkpoint_sha256']:
                    error[0] = 'Checkpoint hash differs from frozen request'
            except Exception as exc: error[0] = repr(exc)
        dist.broadcast_object_list(error, src=0)
        if error[0] is not None: raise ValueError(error[0])
        prompts = []
        if not len(request['prompt_paths']) == len(request['prompts']) == len(request['prompt_records']):
            raise ValueError('Prompt records are incomplete')
        for path, text, rec in zip(request['prompt_paths'], request['prompts'], request['prompt_records']):
            value = torch.load(path, map_location='cpu', weights_only=True)
            if value['prompt'] != text or file_sha(path) != rec['file_sha256'] or tensor_sha(value['prompt_embeds']) != rec['tensor_sha256']:
                raise ValueError('Prompt encoding changed')
            prompts.append(value)
        # Rank zero owns VAE/output; every rank participates in model FSDP.
        session = load_session(request['weights_spec'], device, dense=True,
                               load_decoders=rank == 0, stage1b_current_only=True)
        install_stage1b_processors(session['model'])
        from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
        # Imports the existing published causal module without duplicating KV storage.
        from h3_stage1_clock_probe import tensor_sha as _clock_sha
        from h3_worldattention_causal import VideoKVBank
        context_weight = session['model'].transformer.context_embedder.weight
        context_mesh = context_weight.device_mesh if isinstance(context_weight, DTensor) else None
        total_latents, total_frames = 422, 1433
        total_audio = round(total_frames * 40 / 24)
        blocks = request['blocks']
        if len(blocks) != 84: raise ValueError('Exactly 84 blocks required')
        for ci, row in enumerate(blocks):
            start = 0 if ci == 0 else 2 + 5 * ci
            if row['local_latent_start'] != start or row['latent_count'] != (7 if ci == 0 else 5):
                raise ValueError('Request has invalid startup7/regular5 geometry')
        prompt_indices = [int(row['prompt_index']) for row in blocks]
        if any(i < 0 or i >= len(prompts) for i in prompt_indices): raise ValueError('Unknown prompt index')
        switches = sum(a != b for a, b in zip(prompt_indices, prompt_indices[1:]))
        video_template = torch.zeros(24, total_latents, 48, 84, dtype=torch.bfloat16)
        audio_template = torch.zeros(2, 32, total_audio, dtype=torch.bfloat16)
        selected_names = request.get('selected_policy_names', [p['name'] for p in expected_policies])
        if not selected_names or len(set(selected_names)) != len(selected_names) or any(
                name not in {p['name'] for p in expected_policies} for name in selected_names):
            raise ValueError('Unknown or duplicate selected policy')
        selected_policies = [p for p in expected_policies if p['name'] in selected_names]
        reference_noise = []
        external_reference = request.get('matched_input_reference_timing')
        if external_reference:
            if file_sha(external_reference) != request['matched_input_reference_timing_sha256']:
                raise ValueError('Reused input-noise reference changed')
            reference_records = [json.loads(s) for s in Path(external_reference).read_text().splitlines() if s.strip()]
            if len(reference_records) != 84: raise ValueError('Need 84 matched reference blocks')
            reference_noise = [r['noise_and_position_hashes'] for r in reference_records]
        elif 'current_native' not in selected_names:
            raise ValueError('A history-only retry requires the saved current-only input hashes')
        for policy in selected_policies:
            name = policy['name']; generator = torch.Generator(device=device).manual_seed(int(request['seed']))
            bank = VideoKVBank(local_frames=9, sink_frames=3)
            out = Path(os.environ['INFER_OUTPUT_DIR']) / name
            if rank == 0:
                out.mkdir(parents=True, exist_ok=True)
                (out / 'RESOLVED_CONFIG.json').write_text(json.dumps(dict(request=request,
                    policy=policy, text_timestep='native_video_time', video_reads_history=True,
                    clean_writer_text_reads_history=False, text_KV_committed=False,
                    changed_interface_experiment=policy['text_reads_history'],
                    worker_sha256=file_sha(__file__)), indent=2) + '\n')
            records, archives = [], []; started = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode():
                for ci in range(84):
                    start = 0 if ci == 0 else 2 + 5 * ci; end = start + (7 if ci == 0 else 5)
                    audio_end = round((5 + 17 * ((end - 2) // 5)) * 40 / 24)
                    prompt = prompts[prompt_indices[ci]]; tick_block = time.monotonic()
                    batch = stage1b_chunk(video_template, audio_template, prompt['prompt_embeds'], prompt['text_token_tags'], generator, device, start, end, audio_end)
                    x = batch['inputs']
                    positions_sha = tensor_sha(x['position_ids'])
                    prompt_sha = tensor_sha(x['encoder_hidden_states'])
                    if context_mesh is not None:
                        x['encoder_hidden_states'] = distribute_tensor(x['encoder_hidden_states'], context_mesh, [Replicate()])
                    video = torch.randn(x['hidden_states'].shape, device=device, generator=generator)
                    audio = torch.randn(x['audio_hidden_states'].shape, device=device, generator=generator)
                    hashes = [tensor_sha(video), tensor_sha(audio), positions_sha, prompt_sha]
                    before = stage1b_kv_samples(bank); step_times = []
                    for i in range(4):
                        tv, ta = stage1b_clocks(1 - i / 4)
                        x['hidden_states'], x['audio_hidden_states'] = video, audio; stage1b_clock(batch, tv, ta)
                        tick = time.monotonic()
                        pv, pa = session['model'].transformer(**x, attention_mask=stage1b_context(
                            batch, ci, start, end, bank, text_reads_history=policy['text_reads_history']))
                        torch.cuda.synchronize(device); step_times.append(time.monotonic() - tick)
                        clean_v = (video + (1 - tv) * pv.float()).to(torch.bfloat16)
                        clean_a = (audio + (1 - ta) * pa.float()).to(torch.bfloat16)
                        if i < 3:
                            nv, na = stage1b_clocks(1 - (i + 1) / 4)
                            ev = torch.randn(clean_v.shape, device=device, generator=generator)
                            ea = torch.randn(clean_a.shape, device=device, generator=generator)
                            hashes.extend([tensor_sha(ev), tensor_sha(ea)])
                            # Exact training formula including BF16 multiplication rounding.
                            video = (nv * clean_v + (1 - nv) * ev).to(torch.bfloat16)
                            audio = (na * clean_a + (1 - na) * ea).to(torch.bfloat16)
                        else: video, audio = clean_v, clean_a
                    if not external_reference and name == 'current_native': reference_noise.append(hashes)
                    elif hashes != reference_noise[ci]: raise ValueError('Paired noise/positions/text differ')
                    retained = stage1b_check_samples(before, bank)
                    stage1b_clock(batch, 1., 1.); x['hidden_states'], x['audio_hidden_states'] = video.detach(), audio.detach()
                    tick = time.monotonic()
                    # Keep trained writer identical between the sampling intervention lanes.
                    session['model'].transformer(**x, attention_mask=stage1b_context(batch, ci, start, end, bank, bank))
                    torch.cuda.synchronize(device); commit_seconds = time.monotonic() - tick
                    retained_after = stage1b_check_samples(before, bank)
                    if not bool(torch.isfinite(video).all() and torch.isfinite(audio).all()): raise ValueError('Nonfinite rollout')
                    if rank == 0:
                        archive = out / f'chunk{ci:03d}_GENERATED.pt'
                        torch.save(dict(video=video.cpu(), audio=audio.cpu(), chunk=ci), archive)
                        archives.append(str(archive))
                        rec = dict(chunk=ci, prompt_index=prompt_indices[ci], prompt_switch=ci > 0 and prompt_indices[ci] != prompt_indices[ci-1],
                            start_latent=start, end_latent=end, finite=True, NFE=4, clean_writes=1,
                            old_history_queries_recomputed=0, video_reads_history=True,
                            text_reads_history=policy['text_reads_history'], clean_writer_text_reads_history=False,
                            KV_bytes=bank.bytes, retained_KV_samples_equal=True,
                            KV_sample_rows_checked=[retained, retained_after], KV_sample_scope='two dimensions of head0, retained rows; not full checksum',
                            video_sha256=tensor_sha(video), audio_sha256=tensor_sha(audio), noise_and_position_hashes=hashes,
                            per_step_seconds=step_times, clean_commit_seconds=commit_seconds,
                            wall_seconds=time.monotonic() - tick_block)
                        records.append(rec)
                        with (out / 'TIMING.jsonl').open('a') as stream: stream.write(json.dumps(rec) + '\n')
                        print(name, ci, 'COMPLETE', rec['wall_seconds'], flush=True)
                    del before, batch, x, video, audio, clean_v, clean_a, pv, pa
                # Model cache lifecycle is separate from the decoder. Finished
                # rollout releases its bank before rank-zero bounded VAE decode.
                del bank; torch.cuda.empty_cache(); dist.barrier()
                export_error = [None]
                if rank == 0:
                  try:
                    media = stage1b_decode_long(session, out, archives, total_frames, device)
                    result = dict(status='stage1b_84chunk_complete_visual_review_pending', checkpoint_step=session['checkpoint_step'],
                        checkpoint_sha256=request['weights_spec']['checkpoint_sha256'], request_sha256=file_sha(request_path),
                        training_contract='STAGE1B_CURRENT_ONLY_V1', policy=policy, changed_interface_experiment=policy['text_reads_history'],
                        frames=total_frames, FPS=24, NFE=4, chunks=84, prompt_switches=switches,
                        records=records, **media, old_history_queries_recomputed=0,
                        quality_verified=False, formal_benchmark_admission=False,
                        wall_seconds=time.monotonic() - started, model_load_seconds=session['load_seconds'],
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, GPU=torch.cuda.get_device_name(device))
                    (out / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
                  except Exception as exc:
                    export_error[0] = f'{type(exc).__name__}: {exc}'
                    (out / 'EXPORT_ERROR.json').write_text(json.dumps(dict(error=export_error[0],
                        completed_blocks=len(records), history_recomputation=False), indent=2) + '\n')
                dist.broadcast_object_list(export_error, src=0)
                if export_error[0] is not None: raise RuntimeError(export_error[0])
                dist.barrier()
        if rank == 0:
            (Path(os.environ['INFER_OUTPUT_DIR']) / 'LONG_RUN_SUMMARY.json').write_text(json.dumps(dict(
                status='complete_visual_review_pending', policies=selected_policies, request_sha256=file_sha(request_path)), indent=2) + '\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()


def dense_oracle_short(request_path, request):
    """Native dense H3 4-chunk oracle, no historical KV and no causal mask.

    The A200 chunk files provide the frozen CUDA RNG state before each chunk;
    the batch layout is rebuilt independently and the regenerated noise hashes
    are checked against those records. This is an oracle/reference track, not
    a streaming-memory quality score.
    """
    if not (request.get('diagnostic_only') is True and request.get('mode') == 'dense_oracle_short'
            and request.get('NFE') == 50 and request.get('chunks') == 4):
        raise ValueError('Use the explicit dense oracle four-chunk contract')
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local); torch.set_num_threads(2); device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=40), device_id=device)
    try:
        if dist.get_world_size() != 8: raise ValueError('Dense oracle requires eight ranks')
        identity = [None] * 8; dist.all_gather_object(identity, file_sha(request_path))
        if len(set(identity)) != 1: raise ValueError('Mixed dense oracle requests')
        prompt_values = []
        for path, text, rec in zip(request['prompt_paths'], request['prompts'], request['prompt_records']):
            value = torch.load(path, map_location='cpu', weights_only=True)
            if value['prompt'] != text or file_sha(path) != rec['file_sha256'] or tensor_sha(value['prompt_embeds']) != rec['tensor_sha256']:
                raise ValueError('Dense oracle prompt changed')
            prompt_values.append(value)
        session = load_session(request['weights_spec'], device, dense=True,
                               diagnostic_native_rules=True)
        video_template = torch.zeros(24, 22, 48, 84, dtype=torch.bfloat16)
        audio_template = torch.zeros(2, 32, 122, dtype=torch.bfloat16)
        source_dir = Path(request['noise_reference_dir'])
        records, videos, audios = [], [], []
        started = time.monotonic()
        with torch.inference_mode():
            for ci, row in enumerate(request['blocks']):
                # Use a throw-away RNG for layout construction: matched_spatial_batch
                # only needs its random values for training targets, which are removed.
                dummy = torch.Generator(device=device).manual_seed(92000 + ci)
                start, end = int(row['local_latent_start']), int(row['local_latent_start'] + row['latent_count'])
                batch = stage1b_chunk(video_template, audio_template,
                                      prompt_values[row['prompt_index']]['prompt_embeds'],
                                      prompt_values[row['prompt_index']]['text_token_tags'], dummy, device,
                                      start, end, int(row['audio_end']))
                x = batch['inputs']; ref = torch.load(source_dir / f'chunk{ci:03d}_GENERATED.pt', map_location='cpu', weights_only=True)
                state = ref['generator_before'].detach().cpu().to(torch.uint8)
                generator = torch.Generator(device=device); generator.set_state(state)
                video = torch.randn(x['hidden_states'].shape, device=device, generator=generator)
                audio = torch.randn(x['audio_hidden_states'].shape, device=device, generator=generator)
                initial_hashes=[tensor_sha(video), tensor_sha(audio)]
                expected=ref['initial_noise'];
                if initial_hashes != [expected['video_sha256'], expected['audio_sha256']]:
                    raise ValueError('Dense oracle noise does not match A200 frozen chunk state')
                step_times=[]
                for i in range(50):
                    tv, ta = stage1b_clocks(1 - i / 50); nv, na = stage1b_clocks(1 - (i + 1) / 50)
                    x['hidden_states'], x['audio_hidden_states'] = video, audio; stage1b_clock(batch, tv, ta)
                    tick=time.monotonic(); pv, pa=session['model'].transformer(**x, attention_mask=None); torch.cuda.synchronize(device); step_times.append(time.monotonic()-tick)
                    video=(video.float() + (nv-tv)*pv.float()).to(torch.bfloat16)
                    audio=(audio.float() + (na-ta)*pa.float()).to(torch.bfloat16)
                if not bool(torch.isfinite(video).all() and torch.isfinite(audio).all()): raise ValueError('Dense oracle nonfinite')
                videos.append(video.cpu()); audios.append(audio.cpu())
                records.append(dict(chunk=ci,prompt_index=row['prompt_index'],start_latent=start,end_latent=end,finite=True,NFE=50,initial_video_sha256=initial_hashes[0],initial_audio_sha256=initial_hashes[1],old_history_queries_recomputed=0,KV_bytes=0,per_step_seconds=step_times,wall_seconds=sum(step_times)))
        if rank == 0:
            from sample_demo import unpack_video
            from src.inference.render import decode_and_save
            decoded_video=unpack_video(torch.cat(videos,1)[0],22,48,84).to(device)
            decoded_audio=torch.cat([v[0].reshape(2,-1,32) for v in audios],1).permute(0,2,1).to(device)
            out=Path(os.environ['INFER_OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);archive=out/'DENSE_ORACLE_LATENTS.pt';path=out/'DENSE_ORACLE.mp4'
            torch.save(dict(video=decoded_video.cpu(),audio=decoded_audio.cpu(),records=records),archive)
            session['vae'].to(device);session['audio_vae'].to(device);decode_and_save(decoded_video,decoded_audio,session['vae'],session['audio_vae'],str(path),str(device))
            result=dict(status='dense_oracle_4chunk_complete_reference_only',checkpoint_step=session['checkpoint_step'],checkpoint_sha256=request['weights_spec'].get('checkpoint_sha256'),contract=request['weights_spec']['attention_contract'],frames=73,FPS=24,NFE=50,chunks=4,records=records,video=str(path),video_sha256=file_sha(path),latent_archive=str(archive),latent_archive_sha256=file_sha(archive),old_history_queries_recomputed=0,KV_bytes=0,quality_verified=False,formal_benchmark_admission=False,wall_seconds=time.monotonic()-started,GPU=torch.cuda.get_device_name(device))
            (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
        dist.barrier()
    finally: dist.destroy_process_group()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


_CURRENT_TEXT_CONTEXT_TYPES = {}


def current_text_context(core, context, *, allow_clean_current=False, allow_history=False):
    """Explicit sampling/clean-writer diagnostic; production defaults unchanged."""
    if context.write_bank is not None:
        if not allow_clean_current:
            return context
        if any(s.role == 'noisy' for s in context.splits):
            raise ValueError('A clean-writer intervention must not contain a noisy target')

    context_type = _CURRENT_TEXT_CONTEXT_TYPES.get(core.Context)
    if context_type is None:
        class CurrentTextContext(core.Context):
            def visible_splits(self, query):
                visible = super().visible_splits(query)
                if query.role != 'text':
                    return visible
                role = 'clean' if self.write_bank is not None else 'noisy'
                return tuple(s for s in self.splits if s in visible or
                             (s.role == role and s.block == query.block))
        context_type = _CURRENT_TEXT_CONTEXT_TYPES[core.Context] = CurrentTextContext
    return context_type(context.splits, context.text_reads_history or allow_history,
                        context.cached, context.write_bank, context.modulation_segments)


class CurrentTextAttentionProbe:
    """Delegate QKV/cache arithmetic to the unchanged shared attention processor."""
    def __init__(self, original, core):
        self.original, self.core = original, core
        self.allow_current = False
        self.allow_history = False
        self.allow_clean_current = False
        self.modified_sample_calls = 0
        self.unchanged_commit_calls = 0
        self.modified_commit_calls = 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None, is_causal=False):
        if attention_mask.write_bank is not None:
            if self.allow_clean_current:
                attention_mask = current_text_context(self.core, attention_mask, allow_clean_current=True, allow_history=self.allow_history)
                self.modified_commit_calls += 1
            else:
                self.unchanged_commit_calls += 1
        elif self.allow_current:
            attention_mask = current_text_context(self.core, attention_mask, allow_history=self.allow_history)
            self.modified_sample_calls += 1
        return self.original(attn, hidden_states, rotary_emb, attention_mask, is_causal)


def sampling_text_clock(inputs, context, text_time):
    if context.write_bank is not None or text_time != 'native':
        return inputs
    changed = dict(inputs)
    clock = inputs['timestep'][inputs['timestep_indices']].clone()
    clock[inputs['text_indices']] = clock[inputs['video_indices'][0]].clone()
    changed['timestep'], changed['timestep_indices'] = torch.unique(clock, sorted=True, return_inverse=True)
    return changed


def sampled_bank_state(bank):
    """Small endpoint samples, not a full-KV checksum or history re-encoding."""
    ids, values = [], []
    for layer, entries in sorted(bank.snapshot().items()):
        for entry in entries:
            ids.append((layer, entry.block))
            values.append(torch.cat((entry.key[0, 0, 0, :8].float(),
                                     entry.key[0, -1, -1, -8:].float(),
                                     entry.value[0, 0, 0, :8].float(),
                                     entry.value[0, -1, -1, -8:].float())))
    if not values:
        return {}
    samples = torch.stack(values).cpu()
    return {identity: sample.clone() for identity, sample in zip(ids, samples)}


def streaming_text_probe(request_path, request):
    """Bounded rollouts; reuse FixedKVStream and the verified decoder."""
    import subprocess
    import wave
    from h3_shared_kv_video import Observer, block_template
    from h3_shared_stage1 import TransformerCallProxy
    from h3_shared_kv_heads import require_same_request
    from h3_buffered_video_decode import BufferedVideoDecoder
    from h3_frame_writer import FrameWriter
    from sample_demo import unpack_video

    if not request.get('diagnostic_only') or request.get('strategy') != 'A':
        raise ValueError('Explicit A-rule changed-interface diagnosis required')
    if request.get('NFE') != 50 or request.get('history_seconds') != 8.:
        raise ValueError('Do not change NFE or the committed history window')
    policies = request['policies']
    allowed = {('joint_current', 'fixed_clean'), ('readonly_text', 'native'), ('joint_current', 'native'), ('joint_current_history', 'native')}
    original_group = ({(p['visibility'], p['text_time']) for p in policies} == allowed
                      and len(policies) == 4 and not any(p.get('clean_feedback') for p in policies))
    writer_group = (request.get('clean_writer_diagnostic') is True and len(policies) == 1
                    and policies[0]['visibility'] == 'joint_current' and policies[0]['text_time'] == 'native'
                    and policies[0].get('clean_feedback') is True)
    if not (original_group or writer_group):
        raise ValueError('Use the three sampling controls or one explicit clean-writer control')
    execution_contract = ('diagnostic_current_clean_text_feedback_v1' if writer_group
                          else 'diagnostic_current_text_sampling_only_v1')
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30), device_id=device)
    writer = None
    try:
        if dist.get_world_size() != 8:
            raise ValueError('One eight-card same-video group required')
        require_same_request(file_sha(request_path))
        code, source = shared_modules()
        session = load_session(request['weights_spec'], device)
        core = code['h3_fixed_kv_stage1']
        probes = []
        for block in session['model'].transformer.transformer_blocks:
            while hasattr(block, '_checkpoint_wrapped_module'):
                block = block._checkpoint_wrapped_module
            probe = CurrentTextAttentionProbe(block.attn.processor, core)
            block.attn.set_processor(probe)
            probes.append(probe)
        prompts = [torch.load(p, map_location='cpu', weights_only=True) for p in request['prompt_paths']]
        for p, text in zip(prompts, request['prompts']):
            if p['prompt'] != text:
                raise ValueError('Encoded text differs from the frozen request')
        reference = None
        if request.get('reference_result'):
            reference = json.loads(Path(request['reference_result']).read_text())
            if file_sha(request['reference_result']) != request['reference_result_sha256']:
                raise ValueError('Reference rollout result changed')
        origin = len(prompts[0]['prompt_embeds'])
        for policy in policies:
            for probe in probes:
                probe.allow_current = policy['visibility'] in ('joint_current', 'joint_current_history')
                probe.allow_history = policy['visibility'] == 'joint_current_history'
                probe.allow_clean_current = policy.get('clean_feedback', False)
                probe.modified_sample_calls = probe.unchanged_commit_calls = 0
                probe.modified_commit_calls = 0
            observer = Observer()

            class SamplingProxy(TransformerCallProxy):
                def __call__(self, *, attention_mask, **inputs):
                    if attention_mask.write_bank is not None and policy.get('clean_feedback'):
                        if inputs['audio_indices'].numel():
                            raise ValueError('No audio may enter the clean writer input')
                        times = inputs['timestep'][inputs['timestep_indices']]
                        if not bool((times == 1.).all()):
                            raise ValueError('Clean writer must keep the fixed clean timestep')
                    inputs = sampling_text_clock(inputs, attention_mask, policy['text_time'])
                    return super().__call__(attention_mask=attention_mask, **inputs)

            proxy = SamplingProxy(session['model'], observer)
            stream = code['h3_fixed_kv_runtime'].FixedKVStream(proxy, 'A')
            generator = torch.Generator(device=device).manual_seed(request['seed'])
            out = Path(os.environ['INFER_OUTPUT_DIR']) / policy['name']
            out.mkdir(exist_ok=True)
            decoder = BufferedVideoDecoder(session['vae'], device) if rank == 0 else None
            writer = FrameWriter(out / 'VIDEO.mp4', 24, encoder_threads=2) if rank == 0 else None
            records, audio_pieces = [], []
            started = time.monotonic()
            if rank == 0:
                (out / 'RESOLVED_CONFIG.json').write_text(json.dumps(dict(
                    request=request, policy=policy, shared_source=source,
                    checkpoint_contract_verified=True, execution_contract=execution_contract,
                    shared_production_mask_source_modified=False,
                    clean_commit_rule_unchanged=not policy.get('clean_feedback', False),
                    changed_interface_experiment=True,
                    model_history_recomputation_allowed=False, formal_memory_quality_claim=False), indent=2)+'\n')
            with torch.inference_mode():
                for row in request['blocks']:
                    before = sampled_bank_state(stream.bank)
                    observer.begin(row['block'])
                    rng = generator.get_state().cpu()
                    require_same_request((row['block'], tensor_sha(rng)))
                    block = block_template(row, prompts[row['prompt_index']], origin)
                    tick = time.monotonic()
                    video, audio, stats = stream.generate(block, generator, device, 50)
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - tick
                    timing = observer.timings()
                    after = sampled_bank_state(stream.bank)
                    retained = before.keys() & after.keys()
                    unchanged = all(torch.equal(before[key], after[key]) for key in retained)
                    ok = torch.tensor(int(unchanged), device=device)
                    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
                    if not ok.item():
                        raise ValueError('Previously committed retained KV samples changed')
                    if reference is not None:
                        for key in ('video_sha256', 'audio_sha256'):
                            if observer.initial_noise[key] != reference['records'][row['block']][key]:
                                raise ValueError('Actual initial noise differs from retained production control')
                    if rank == 0:
                        latent_count = int(row.get('latent_count', 7 if int(row['block']) == 0 else 5))
                        latent = unpack_video(video, latent_count, 48, 84).to(torch.bfloat16)
                        sound = audio[0].reshape(2, -1, 32).permute(0, 2, 1).to(torch.bfloat16)
                        torch.save(dict(video=latent.cpu(), audio=sound.cpu(), generator_before=rng,
                                        generator_after=generator.get_state().cpu(), row=row,
                                        initial_noise=observer.initial_noise), out / f'chunk{row["block"]:03d}_GENERATED.pt')
                        if session['reshard']:
                            session['vae'].to(device)
                        decoded = decoder.push(latent)
                        if decoded is not None:
                            writer.append(decoded)
                        if session['reshard']:
                            session['vae'].cpu()
                            torch.cuda.empty_cache()
                        audio_pieces.append(sound)
                        record = dict(block=row['block'], prompt_index=row['prompt_index'],
                                      sample_wall_seconds=elapsed, **timing, **observer.initial_noise,
                                      history_blocks=list(stats['read_past_blocks']), KV_bytes_per_rank=stats['KV_bytes'],
                                      old_history_queries_recomputed=0, retained_KV_endpoint_samples_exact=True,
                                      retained_KV_sample_scope='32 endpoint values per layer/block; not full checksum')
                        records.append(record)
                        with (out / 'TIMING.jsonl').open('a') as f:
                            f.write(json.dumps(record)+'\n')
                        print(policy['name'], row['block'], 'COMPLETE', elapsed, flush=True)
                    dist.barrier()
                counts = dict(modified_sample_attention_calls=probes[0].modified_sample_calls,
                              unchanged_commit_attention_calls=probes[0].unchanged_commit_calls,
                              modified_clean_commit_attention_calls=probes[0].modified_commit_calls)
                expected_clean_modified = len(request['blocks']) if policy.get('clean_feedback') else 0
                if (counts['modified_clean_commit_attention_calls'] != expected_clean_modified or
                        counts['unchanged_commit_attention_calls'] != len(request['blocks'])-expected_clean_modified):
                    raise ValueError('Clean-commit lifecycle changed')
                expected_modified = 50 * len(request['blocks']) if policy['visibility'] in ('joint_current', 'joint_current_history') else 0
                if counts['modified_sample_attention_calls'] != expected_modified:
                    raise ValueError('Current-text attention intervention did not execute as declared')
                if rank == 0:
                    if session['reshard']:
                        session['vae'].to(device)
                    tail = decoder.finish()
                    if tail is not None:
                        writer.append(tail)
                    writer.close()
                    if writer.frames != request['actual_frames']:
                        raise ValueError('Diagnostic output duration differs from fixed control')
                    if session['reshard']:
                        session['vae'].cpu()
                        session['audio_vae'].to(device)
                        torch.cuda.empty_cache()
                    avae = session['audio_vae']
                    mean = torch.tensor(avae.config.latents_mean, device=device).view(1, -1, 1)
                    std = torch.tensor(avae.config.latents_std, device=device).view(1, -1, 1)
                    pcm = avae.decode(torch.cat(audio_pieces, 2)*std+mean, return_dict=False)[0].float().permute(1,0,2)[0]
                    wav = out / 'AUDIO.wav'
                    with wave.open(str(wav), 'wb') as f:
                        f.setnchannels(2); f.setsampwidth(2); f.setframerate(32000)
                        f.writeframes((pcm.clamp(-1,1).cpu().T*32767).round().short().contiguous().numpy().tobytes())
                    import imageio_ffmpeg
                    mux = out / 'MUX.mp4'
                    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v','error','-nostdin','-y',
                                    '-i',str(out/'VIDEO.mp4'),'-i',str(wav),'-c:v','copy','-c:a','aac',
                                    '-threads','2','-af','apad','-t',str(writer.frames/24),str(mux)],
                                   check=True, timeout=180)
                    mux.replace(out/'VIDEO.mp4')
                    result = dict(status='complete_changed_interface_streaming_diagnostic', policy=policy,
                                  actual_frames=writer.frames, FPS=24, NFE=50, records=records,
                                  video=str(out/'VIDEO.mp4'), video_sha256=file_sha(out/'VIDEO.mp4'),
                                  shared_source=source, checkpoint_sha256=request['weights_spec']['checkpoint_sha256'],
                                  request_sha256=file_sha(request_path), **counts,
                                  clean_commit_rule_unchanged=not policy.get('clean_feedback', False),
                                  clean_commit_rule='own_clean_video_text_feedback' if policy.get('clean_feedback') else 'shared_A_writer',
                                  changed_interface_experiment=True, execution_contract=execution_contract,
                                  old_history_queries_recomputed=0,
                                  model_load_seconds=session['load_seconds'],
                                  model_load_accounting='shared across declared rollouts, do not sum per-policy',
                                  wall_seconds=time.monotonic()-started, formal_memory_quality_claim=False)
                    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
                    if session['reshard']:
                        session['audio_vae'].cpu()
                        torch.cuda.empty_cache()
                dist.barrier()
            peaks = [None] * 8
            dist.all_gather_object(peaks, torch.cuda.max_memory_allocated()/2**30)
            if rank == 0:
                (Path(os.environ['INFER_OUTPUT_DIR'])/'RUN_SUMMARY.json').write_text(json.dumps(dict(
                    policies=policies, load_seconds=session['load_seconds'], peak_allocated_gib_by_rank=peaks,
                    formal_memory_quality_claim=False, execution_contract=execution_contract),indent=2)+'\n')
    finally:
        if writer is not None and not writer.closed:
            writer.__exit__(RuntimeError, None, None)
        dist.destroy_process_group()


class RuleProbe:
    def __init__(self, model, core, visibility, text_time):
        self.model, self.core = model, core
        self.visibility, self.text_time = visibility, text_time
        self.calls = 0
        self.first = None

    def __getattr__(self, name):
        try:
            return getattr(self.model, name)
        except AttributeError:
            return getattr(self.model.transformer, name)

    def __call__(self, **inputs):
        inputs = dict(inputs)
        if self.text_time == 'fixed_clean':
            clock = inputs['timestep'][inputs['timestep_indices']].clone()
            clock[inputs['text_indices']] = 1.
            inputs['timestep'], inputs['timestep_indices'] = torch.unique(clock, sorted=True, return_inverse=True)
        if self.visibility == 'readonly_text':
            nt = inputs['text_indices'].numel()
            total = len(inputs['position_ids'])
            ctx = self.core.Context((self.core.Split('text', 0, 0, nt, ()),
                                     self.core.Split('noisy', 0, nt, total, ())))
            inputs['attention_mask'] = self.core.bind(self.model.transformer, inputs, ctx)
        output = self.model(**inputs)
        if self.calls == 0:
            negative = self.model(**{**inputs, 'encoder_hidden_states': torch.zeros_like(inputs['encoder_hidden_states'])})
            self.first = dict(prompt_tensor_sha256=tensor_sha(inputs['encoder_hidden_states']),
                              video_noise_sha256=tensor_sha(inputs['hidden_states']),
                              audio_noise_sha256=tensor_sha(inputs['audio_hidden_states']),
                              positions_sha256=tensor_sha(inputs['position_ids']),
                              actual_text_time=float(inputs['timestep'][inputs['timestep_indices'][inputs['text_indices'][0]]]),
                              video_velocity_sha256=tensor_sha(output[0]),
                              audio_velocity_sha256=tensor_sha(output[1]),
                              zero_text_video_relative_l2=float(torch.linalg.vector_norm(output[0].float()-negative[0].float()) /
                                                               torch.linalg.vector_norm(output[0].float()).clamp_min(1e-12)))
        self.calls += 1
        return output


def legacy_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    args = ap.parse_args()
    request_path = Path(args.request)
    request = json.loads(request_path.read_text())
    if request.get('mode') == 'streaming_text_probe':
        streaming_text_probe(request_path, request)
        return
    if request.get('diagnostic_only') is not True or request.get('history_blocks') != 0:
        raise ValueError('Explicit empty-history diagnostic required')
    if request.get('NFE') != 50 or request.get('num_frames') != 124:
        raise ValueError('Use the declared unchanged 124-frame, 50-NFE short protocol')
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30), device_id=device)
    try:
        if dist.get_world_size() != 8:
            raise ValueError('One eight-rank model session is required')
        # Prevent accidentally assembling one model from different weight shards.
        identities = [None] * 8
        dist.all_gather_object(identities, file_sha(request_path))
        if len(set(identities)) != 1:
            raise ValueError('All ranks must load the same frozen checkpoint request')
        if request['weights_spec']['kind'] == 'full':
            # Read the large shared checkpoint only once for the hash audit.
            error = [None]
            if rank == 0:
                try:
                    if file_sha(request['weights_spec']['checkpoint']) != request['weights_spec']['checkpoint_sha256']:
                        error[0] = 'Checkpoint file hash differs from the frozen handoff'
                except Exception as exc:
                    error[0] = repr(exc)
            dist.broadcast_object_list(error, src=0)
            if error[0] is not None:
                raise ValueError(error[0])
        code, source = shared_modules()
        core = code['h3_fixed_kv_stage1']
        session = load_session(request['weights_spec'], device, dense=True, diagnostic_native_rules=True)
        visibility, text_time = [('joint', 'native'), ('joint', 'fixed_clean'),
                                  ('readonly_text', 'native'), ('readonly_text', 'fixed_clean')][rank % 4]
        if visibility == 'readonly_text':
            blocks = []
            for block in session['model'].transformer.transformer_blocks:
                while hasattr(block, '_checkpoint_wrapped_module'):
                    block = block._checkpoint_wrapped_module
                blocks.append(block)
            core.install(SimpleNamespace(transformer_blocks=blocks, token_refiner=session['model'].transformer.token_refiner))
        proxy = RuleProbe(session['model'], core, visibility, text_time)
        prompt = torch.load(request['prompt_path'], map_location='cpu', weights_only=True)
        if prompt['prompt'] != request['prompt']:
            raise ValueError('Prompt plaintext does not match its encoding')
        if tensor_sha(prompt['prompt_embeds']) != request['prompt_tensor_sha256']:
            raise ValueError('Prompt tensor changed')
        seed = request['seed'] + rank // 4
        times = []
        started = time.monotonic()
        with torch.inference_mode():
            video, audio = generate_latents(proxy, prompt['prompt_embeds'].to(device), prompt['text_token_tags'],
                                           124, 50, seed, device, step_seconds=times)
            if len(times) != 50 or proxy.calls != 50:
                raise ValueError('Sampler step count changed')
            if not bool(torch.isfinite(video).all() and torch.isfinite(audio).all()):
                raise ValueError('Nonfinite generated output')
            if proxy.first['prompt_tensor_sha256'] != request['prompt_tensor_sha256']:
                raise ValueError('Actual model input differs from frozen text tensor')
            out = Path(os.environ['INFER_OUTPUT_DIR'])
            name = f'rank{rank}_{visibility}_{text_time}_seed{seed}'
            archive = out / (name + '_LATENTS.pt')
            torch.save(dict(video=video.cpu(), audio=audio.cpu(), first_NFE=proxy.first), archive)
            path = out / (name + '.mp4')
            session['vae'].to(device)
            session['audio_vae'].to(device)
            decode_and_save(video, audio, session['vae'], session['audio_vae'], str(path), str(device))
        report = dict(status='complete_empty_history_diagnostic', rank=rank, seed=seed,
                      weight_id=request['weight_id'], checkpoint_sha256=request['weights_spec'].get('checkpoint_sha256'),
                      checkpoint_step=session['checkpoint_step'], visibility=visibility, text_time=text_time,
                      current_document_latent_frames=37, frames=124, FPS=24, NFE=50,
                      extra_zero_text_model_evaluations=1, CFG=1., history_blocks=0, prompt_switches=0,
                      first_NFE=proxy.first, video=str(path), video_sha256=file_sha(path),
                      latent_archive=str(archive), latent_archive_sha256=file_sha(archive),
                      request_sha256=file_sha(request_path), shared_source=source['bundle_sha256'],
                      total_seconds=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      diagnostic_only=True, formal_memory_quality_claim=False, changed_interface_experiment=True,
                      frame_window_protocol='native legal 124-frame document; not fixed5 rollout',
                      GPU=torch.cuda.get_device_name(device))
        (out / (name + '_RESULT.json')).write_text(json.dumps(report, indent=2)+'\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    path = Path(ap.parse_args().request)
    request = json.loads(path.read_text())
    if request.get('mode') == 'stage1b_current_only_short':
        return stage1b_short(path, request)
    if request.get('mode') == 'stage1b_current_only_long':
        return stage1b_long(path, request)
    if request.get('mode') == 'dense_oracle_short':
        return dense_oracle_short(path, request)
    return legacy_main()


if __name__ == '__main__':
    main()
