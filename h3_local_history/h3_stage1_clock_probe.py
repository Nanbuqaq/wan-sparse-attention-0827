"""Matched Stage1 ODE-v2 clock diagnosis with immutable, write-once video KV.

Eight ranks share one FSDP checkpoint session, but each produces an independent
condition. No head sharding. Training-side causal attention is frozen verbatim;
only the explicitly requested student text clock changes between paired runs.
"""
import argparse
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from train_h3_v2_local_history_ddp import bootstrap
bootstrap()

FROZEN = Path(__file__).resolve().parent / 'stage1_ode_v2'
sys.path.insert(0, str(FROZEN))
from h3_worldattention_causal import (
    H3CausalContext, PackedCausalLayout, VideoKVBank, install_chunk_causal_attention)
from h3_spatial_history_flow import matched_spatial_batch
from h3_shared_loader import load_session, STAGE1_ODE_V2_CONTRACT

CHUNKS = ((0, 7), (7, 12), (12, 17), (17, 22))


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def verify_frozen():
    manifest = json.loads((FROZEN / 'MANIFEST.json').read_text())
    if manifest['contract'] != STAGE1_ODE_V2_CONTRACT:
        raise ValueError('Unsupported frozen training contract')
    for name, digest in manifest['hashes'].items():
        if file_sha(FROZEN / name) != digest:
            raise ValueError('Frozen common source changed: ' + name)
    for obj in [VideoKVBank, matched_spatial_batch]:
        path = Path(inspect.getfile(obj)).resolve()
        if path.parent != FROZEN.resolve() or file_sha(path) != manifest['hashes'][path.name]:
            raise ValueError('A different common module was already imported: ' + str(path))
    import h3_attention_backend
    from diffusers import MiniMaxH3Transformer3DModel
    for name, path in [('h3_attention_backend.py', h3_attention_backend.__file__),
                       ('native_h3.py', inspect.getfile(MiniMaxH3Transformer3DModel))]:
        if file_sha(path) != manifest['dependency_hashes'][name]:
            raise ValueError('Frozen causal runtime dependency changed: ' + name)
    return manifest


def clocks(u):
    # Exact H3 shifts from the actual aligned teacher/student helper.
    return 1 - 12 * u / (1 + 11 * u), 1 - 3 * u / (1 + 2 * u)


def set_clock(batch, tv, ta, text_clock):
    x = batch['inputs']
    if text_clock not in ('legacy_video_time', 'training_fixed_one'):
        raise ValueError('Unknown text-clock condition')
    times = torch.ones(len(x['position_ids']), device=x['position_ids'].device)
    times[x['video_indices']] = tv
    times[x['audio_indices']] = ta
    if text_clock == 'legacy_video_time':
        times[x['text_indices']] = tv
    x['timestep'], x['timestep_indices'] = torch.unique(times, sorted=True, return_inverse=True)


def chunk_contract(video, audio, prompt, tags, generator, device, start, end, audio_end):
    # Same call and RNG consumption as the SHA-matched Stage1 chunk_contract.
    batch = matched_spatial_batch(video, audio, prompt, tags,
        torch.ones(audio.shape[-1], dtype=torch.bool), generator, device,
        history_frames=0, history_start_latents=0, history_end_latents=0,
        target_start_latents=start, target_end_latents=end,
        target_audio_start_latents=0 if start == 0 else round((5 + 17 * ((start - 2) // 5)) * 40 / 24),
        target_audio_end_latents=audio_end, transfer_used_only=True)
    batch.pop('target_video', None)
    batch.pop('target_audio', None)
    batch['prefix_rows'] = batch['inputs']['text_indices'].numel()
    batch['inputs']['state_hidden_states'] = batch['inputs']['encoder_hidden_states'].new_empty(
        (1, 0, 5376), dtype=torch.bfloat16)
    return batch


def context_for(batch, chunk, start, end, bank, write_bank=None):
    x = batch['inputs']
    blocks = torch.full((len(x['position_ids']),), chunk, device=x['position_ids'].device, dtype=torch.long)
    blocks[x['text_indices']] = -1
    frames = torch.full_like(blocks, -1)
    frames[x['video_indices']] = torch.arange(start, end, device=frames.device).repeat_interleave(1008)
    return H3CausalContext(PackedCausalLayout(blocks, frames, local_frames=12, sink_frames=3),
                           cached=bank.snapshot(), write_bank=write_bank)


def bank_samples(bank):
    # Endpoint values are a limited mutation check, not a full checksum.
    return {layer: ((entry.frames.detach().cpu() * 1008 + torch.arange(entry.frames.numel()) % 1008),
                    torch.cat((entry.key[0, 0, :, :2], entry.value[0, 0, :, :2]), -1).cpu())
            for layer, entry in bank.snapshot().items()}


def require_same_samples(before, after):
    for layer, (frames, values) in before.items():
        new_frames, new_values = after[layer]
        rows = torch.isin(frames, new_frames)
        if not rows.any():
            continue
        ids = torch.searchsorted(new_frames, frames[rows])
        if not torch.equal(values[rows], new_values[ids]):
            raise ValueError('Retained committed KV endpoint values changed')


def generate_chunk(model, batch, chunk, start, end, bank, generator, nfe, device, text_clock):
    x = batch['inputs']
    video = torch.randn(x['hidden_states'].shape, device=device, generator=generator)
    audio = torch.randn(x['audio_hidden_states'].shape, device=device, generator=generator)
    initial = (video.detach().cpu().clone(), audio.detach().cpu().clone())
    before = bank_samples(bank)
    times = []
    first = None
    for i in range(nfe):
        tv, ta = clocks(1 - i / nfe)
        nv, na = clocks(1 - (i + 1) / nfe)
        x['hidden_states'], x['audio_hidden_states'] = video, audio
        set_clock(batch, tv, ta, text_clock)
        ctx = context_for(batch, chunk, start, end, bank)
        if i == 0:
            first = dict(text_sha256=tensor_sha(x['encoder_hidden_states']),
                         positions_sha256=tensor_sha(x['position_ids']),
                         video_noise_sha256=tensor_sha(video), audio_noise_sha256=tensor_sha(audio),
                         timestep_rows_sha256=tensor_sha(x['timestep'][x['timestep_indices']]),
                         text_time=float(x['timestep'][x['timestep_indices'][x['text_indices'][0]]]),
                         video_time=tv, audio_time=ta)
        tick = time.monotonic()
        pv, pa = model(**x, attention_mask=ctx)
        # Match actual legacy Stage1's BF16 round after every Euler update.
        video = (video.float() + (nv - tv) * pv.float()).to(torch.bfloat16)
        audio = (audio.float() + (na - ta) * pa.float()).to(torch.bfloat16)
        torch.cuda.synchronize(device)
        times.append(time.monotonic() - tick)
    require_same_samples(before, bank_samples(bank))
    x['hidden_states'], x['audio_hidden_states'] = video.detach(), audio.detach()
    set_clock(batch, 1., 1., text_clock)
    tick = time.monotonic()
    model(**x, attention_mask=context_for(batch, chunk, start, end, bank, bank))
    torch.cuda.synchronize(device)
    commit_seconds = time.monotonic() - tick
    require_same_samples(before, bank_samples(bank))
    if not bool(torch.isfinite(video).all() and torch.isfinite(audio).all()):
        raise ValueError('Nonfinite Stage1 output')
    return video, audio, initial, dict(chunk=chunk, start=start, end=end, first_NFE=first,
        NFE=nfe, clean_writes=1, old_history_queries_recomputed=0,
        KV_bytes=bank.bytes, retained_KV_endpoint_samples_equal=True,
        KV_check_scope='two dimensions of first head across retained rows; not full checksum',
        per_NFE_seconds=times, clean_commit_seconds=commit_seconds,
        video_sha256=tensor_sha(video), audio_sha256=tensor_sha(audio))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    args = ap.parse_args()
    request = json.loads(Path(args.request).read_text())
    if not (request.get('diagnostic_only') is True and request['NFE'] == 48
            and request['weights_spec']['attention_contract'] == STAGE1_ODE_V2_CONTRACT
            and request.get('history_recomputation_allowed') is False):
        raise ValueError('Use the exact 48-NFE fixed-KV clock diagnosis contract')
    manifest = verify_frozen()
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=40), device_id=device)
    try:
        if dist.get_world_size() != 8 or len(request['conditions']) != 8:
            raise ValueError('Eight independent conditions must share one checkpoint session')
        identities = [None] * 8
        dist.all_gather_object(identities, file_sha(args.request))
        if len(set(identities)) != 1:
            raise ValueError('Mixed checkpoint/session requests')
        condition = request['conditions'][rank]
        prompt = request['prompts'][condition['prompt_index']]
        emb = torch.load(prompt['path'], map_location='cpu', weights_only=True)
        if (emb['prompt'] != prompt['text'] or file_sha(prompt['path']) != prompt['file_sha256']
                or tensor_sha(emb['prompt_embeds']) != prompt['tensor_sha256']):
            raise ValueError('Frozen prompt text/tensor mismatch')
        session = load_session(request['weights_spec'], device, dense=True, stage1_ode_v2=True)
        model = session['model']
        blocks = []
        for block in model.transformer.transformer_blocks:
            while hasattr(block, '_checkpoint_wrapped_module'):
                block = block._checkpoint_wrapped_module
            blocks.append(block)
        from types import SimpleNamespace
        install_chunk_causal_attention(SimpleNamespace(transformer_blocks=blocks))
        for block in blocks:
            block.ff_chunk_size, block.ff_chunk_checkpoint = 8192, False
        generator = torch.Generator(device=device).manual_seed(condition['seed'])
        video_template = torch.zeros(24, 22, 48, 84, dtype=torch.bfloat16)
        audio_template = torch.zeros(2, 32, 122, dtype=torch.bfloat16)
        bank = VideoKVBank(local_frames=12, sink_frames=3)
        records, videos, audios, initials = [], [], [], []
        started = time.monotonic()
        with torch.no_grad():
            for ci, (start, end) in enumerate(CHUNKS):
                a1 = 122 if end == 22 else round((5 + 17 * ((end - 2) // 5)) * 40 / 24)
                rng_before = tensor_sha(generator.get_state())
                batch = chunk_contract(video_template, audio_template, emb['prompt_embeds'],
                    emb['text_token_tags'], generator, device, start, end, a1)
                v, a, noise, rec = generate_chunk(model, batch, ci, start, end, bank, generator,
                    request['NFE'], device, condition['text_clock'])
                rec.update(generator_before=rng_before, generator_after=tensor_sha(generator.get_state()))
                records.append(rec)
                videos.append(v.cpu()); audios.append(a.cpu()); initials.append(noise)
                print(json.dumps(dict(event='chunk_complete', rank=rank, chunk=ci,
                                     KV_bytes=bank.bytes, text_clock=condition['text_clock'])), flush=True)
            # Release 20+GiB/rank bank before VAE, without editing any retained KV.
            del bank, batch, v, a
            torch.cuda.empty_cache()
            # Packed native audio is [channel0 time rows | channel1 time rows].
            from sample_demo import unpack_video
            video_rows = torch.cat(videos, dim=1)
            decoded_video = unpack_video(video_rows[0], 22, 48, 84).to(device)
            decoded_audio = torch.cat([x[0].reshape(2, -1, 32) for x in audios], dim=1).permute(0, 2, 1).to(device)
            out = Path(os.environ['INFER_OUTPUT_DIR'])
            name = f"rank{rank}_{prompt['id']}_{condition['text_clock']}_seed{condition['seed']}"
            archive, path = out / (name + '_LATENTS.pt'), out / (name + '.mp4')
            torch.save(dict(video=decoded_video.cpu(), audio=decoded_audio.cpu(),
                            chunk_video=videos, chunk_audio=audios, initial_noise=initials, records=records), archive)
            from src.inference.render import decode_and_save
            decode_and_save(decoded_video, decoded_audio, session['vae'], session['audio_vae'], str(path), str(device))
        report = dict(status='completed_stage1_clock_diagnostic', condition=condition,
            checkpoint_step=session['checkpoint_step'], checkpoint_sha256=request['weights_spec']['checkpoint_sha256'],
            request_sha256=file_sha(args.request), common_source=manifest,
            attention_contract=STAGE1_ODE_V2_CONTRACT, frames=73, FPS=24, NFE=48, CFG=1.,
            prompt_switches=0, chunk_latents=[7, 5, 5, 5], history_local_frames_including_current=12,
            committed_audio_KV=False, current_audio_tokens_in_video_writer=True,
            no_source_audio=True, audio_future_context_allowed=False, audio_encoder_lookahead_sec=0.,
            current_audio_policy='joint current generated AV block; indirect audio contribution to video KV retained',
            records=records, video=str(path), video_sha256=file_sha(path),
            latent_archive=str(archive), latent_archive_sha256=file_sha(archive),
            load_seconds=session['load_seconds'], generation_and_decode_seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, GPU=torch.cuda.get_device_name(device),
            quality_scored=False, diagnostic_only=True, formal_AR_admission=False)
        (out / (name + '_RESULT.json')).write_text(json.dumps(report, indent=2) + '\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
