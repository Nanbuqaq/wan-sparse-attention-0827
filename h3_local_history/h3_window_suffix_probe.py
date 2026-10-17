"""Replay the same generated suffix with 5+5 or 10 latent target windows.

Two ranks use carry/zero history under one window geometry. Separate sessions
compare window geometries so FSDP collectives always have equal forward counts.
This is an offline generation diagnostic; no training or default policy changes.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import wave

import imageio_ffmpeg
import torch
import torch.distributed as dist

from train_h3_v2_local_history_ddp import bootstrap
bootstrap()
from h3_fixed_chunk_contract import fixed_matched_batch
from h3_fixed_stream_decode import FixedChunkVideoDecoder
from h3_frame_writer import FrameWriter
from h3_generated_video_history import GeneratedVideoHistory
from h3_shared_loader import load_session
from h3_variable_sampling_case import decoder_chunks
from h3_window_noise_contract import combine_noise, joint_row
from sample_demo import unpack_video


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--window', type=int, choices=[5, 10], required=True)
    args = parser.parse_args()
    root, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((root / 'INPUT.json').read_text())
    rank, local, world = [int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')]
    if world != 2:
        raise ValueError('Use two ranks per window session')
    mode = ['carry', 'zero'][rank]
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(2)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=60))
    try:
        for name, expected in meta['input_file_sha256'].items():
            if file_sha(root / name) != expected:
                raise ValueError('Frozen input changed: ' + name)
        case, source, first = meta['case'], meta['source'], meta['first_replayed_chunk']
        original_rows = case['chunks'][first:]
        merged = joint_row(original_rows)
        references = [torch.load(root / f'REFERENCE_{i:03d}.pt', map_location='cpu', weights_only=True)
                      for i in range(first, first + 2)]
        combined_video, combined_audio = combine_noise(references, original_rows)
        prefix = torch.load(root / 'PREFIX.pt', map_location='cpu', weights_only=True)
        start, audio_start = merged['video_latent_start'], merged['audio_latent_start']
        if prefix['completed_NN_chunk'] != first - 1 or prefix['generated_ids'] != list(range(start)):
            raise ValueError('Prefix does not end exactly before the common suffix')
        if file_sha(root / f'PROMPT_{first:03d}.pt') != file_sha(root / f'PROMPT_{first+1:03d}.pt'):
            raise ValueError('Merged suffix does not share an identical prompt')
        rows = original_rows if args.window == 5 else [merged]
        noises = [(r['initial_video_noise'], r['initial_audio_noise']) for r in references]
        if args.window == 10:
            noises = [(combined_video, combined_audio)]
        session = load_session(meta['weights_spec'], device)
        model, processors = session['model'], session['processors']
        for processor in processors:
            processor.text_policy = 'text_full'
        video, audio = prefix['video'].clone(), prefix['audio'].clone()
        video[:, start:] = 0
        audio[:, :, audio_start:] = 0
        history = GeneratedVideoHistory(source)
        history.put(0, video[:, :start])
        history.evict(start)
        generator = torch.Generator(device=device).manual_seed(case['seed'])
        mask = torch.ones(source['audio_latent_count'], dtype=torch.bool)
        records = []
        torch.cuda.reset_peak_memory_stats()
        dist.barrier()
        begin = time.monotonic()
        with torch.inference_mode():
            for number, (row, noise) in enumerate(zip(rows, noises)):
                prompt = torch.load(root / f'PROMPT_{first:03d}.pt', map_location='cpu', weights_only=True)
                chosen, past = history.select(row['video_latent_start'])
                batch = fixed_matched_batch(row, source, video, audio, prompt['prompt_embeds'],
                                            prompt['text_token_tags'], mask, generator, device,
                                            retained_ids=chosen, retained_video=past)
                batch.update(sampling=True, zero_history_state=mode == 'zero')
                inputs = batch['inputs']
                for processor in processors:
                    processor.prefix_rows = batch['prefix_rows']
                    processor.text_rows = int(inputs['text_indices'].numel())
                    processor.prefix_reads_current = not chosen
                xv, xa = [v.to(device).clone() for v in noise]
                if xv.shape != inputs['hidden_states'].shape or xa.shape != inputs['audio_hidden_states'].shape:
                    raise ValueError('Coordinate-mapped noise does not fit the sampler inputs')
                nfe = meta['NFE']
                positions = inputs['position_ids']
                position_hashes = {k: tensor_sha(positions[inputs[k + '_indices']].float())
                                   for k in ('text', 'state', 'audio', 'video')}
                tick = time.monotonic()
                for i in range(nfe):
                    u, unext = 1 - i / nfe, 1 - (i + 1) / nfe
                    tv, ta = 1 - 12 * u / (1 + 11 * u), 1 - 3 * u / (1 + 2 * u)
                    vn, an = 1 - 12 * unext / (1 + 11 * unext), 1 - 3 * unext / (1 + 2 * unext)
                    rt = torch.full((len(positions),), tv, device=device)
                    rt[inputs['state_indices']] = 1.
                    rt[inputs['audio_indices']] = ta
                    inputs['timestep'], inputs['timestep_indices'] = torch.unique(rt, sorted=True, return_inverse=True)
                    inputs['hidden_states'], inputs['audio_hidden_states'] = xv, xa
                    pv, pa = model(batch)
                    xv, xa = xv + (vn - tv) * pv.float(), xa + (an - ta) * pa.float()
                    if i % 10 == 0 or i == nfe - 1:
                        torch.cuda.synchronize()
                        print(json.dumps(dict(event='window_probe_NFE', mode=mode, window=args.window,
                                              chunk=number, NFE=i + 1, seconds=time.monotonic() - tick)), flush=True)
                a, b = row['video_latent_start'], row['video_latent_end']
                aa, ae = row['audio_latent_start'], row['audio_latent_end']
                chunk = unpack_video(xv, b - a, 48, 84).to(torch.bfloat16)[0].cpu()
                generated_audio = xa[0].reshape(2, ae - aa, 32).permute(0, 2, 1).to(torch.bfloat16).cpu()
                if not torch.isfinite(chunk).all() or not torch.isfinite(generated_audio).all():
                    raise ValueError('Nonfinite suffix output')
                video[:, a:b], audio[:, :, aa:ae] = chunk, generated_audio
                records.append(dict(chunk=number, mode=mode, window=args.window, row=row,
                                    history_ids=chosen, history_sha256=tensor_sha(past) if past is not None else None,
                                    zero_history_projection=mode == 'zero', position_hashes_fp32=position_hashes,
                                    initial_video_noise_sha256=tensor_sha(noise[0]),
                                    initial_audio_noise_sha256=tensor_sha(noise[1]),
                                    NFE=nfe, CFG=1., DiT_seconds=time.monotonic() - tick))
                history.put(a, chunk)
                records[-1]['evicted_ids'] = history.evict(b)
            generated = out / (mode + '_GENERATED.pt')
            torch.save(dict(video=video, audio=audio, records=records), generated)
            vae = session['vae']
            vae.to(device)
            decoder = FixedChunkVideoDecoder(vae, device)
            previous = None
            path = out / (mode + '.mp4')
            # Decode both variants with the ORIGINAL five-latent segmentation.
            with FrameWriter(path, case['nominal_output_fps']) as writer:
                for row in case['chunks']:
                    if previous is not None and previous != row['parent_window_id']:
                        tail = decoder.flush()
                        if tail is not None:
                            writer.append(tail.cpu())
                    a, b = row['video_latent_start'], row['video_latent_end']
                    for part in decoder_chunks(video[None, :, a:b].to(device), a):
                        writer.append(decoder.push(part).cpu())
                    previous = row['parent_window_id']
                tail = decoder.flush()
                if tail is not None:
                    writer.append(tail.cpu())
                writer.close()
                artifact, frames = writer.artifact(), writer.frames
            if frames != source['valid_output_frames']:
                raise ValueError('Suffix replay changed the output duration')
            vae.cpu()
            torch.cuda.empty_cache()
            avae = session['audio_vae']
            avae.to(device)
            mean = torch.tensor(avae.config.latents_mean, device=device).view(1, -1, 1)
            std = torch.tensor(avae.config.latents_std, device=device).view(1, -1, 1)
            aw = avae.decode(audio.to(device) * std + mean, return_dict=False)[0].float().permute(1, 0, 2)[0].clamp(-1, 1).cpu()
            wav = path.with_suffix('.wav')
            with wave.open(str(wav), 'wb') as stream:
                stream.setnchannels(2)
                stream.setsampwidth(2)
                stream.setframerate(32000)
                stream.writeframes((aw.T * 32767).round().short().contiguous().numpy().tobytes())
            mux = path.with_suffix('.mux.mp4')
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-nostdin', '-y', '-i', str(path), '-i', str(wav),
                            '-c:v', 'copy', '-c:a', 'aac', '-af', 'apad', '-t', str(frames / case['nominal_output_fps']), str(mux)],
                           check=True, timeout=120)
            os.replace(mux, path)
            artifact.update(sha256=file_sha(path), audio_muxed=True)
        ref = references[-1]
        result = dict(status='complete', mode=mode, window=args.window, weights_spec=meta['weights_spec'],
                      artifact=artifact, generated=str(generated), records=records,
                      prefix_sha256=meta['input_file_sha256']['PREFIX.pt'],
                      input_manifest_sha256=file_sha(root / 'INPUT.json'),
                      original_suffix_video_max_abs=float((video[:, start:].float() - ref['video'][:, start:].float()).abs().max()),
                      original_suffix_audio_max_abs=float((audio[:, :, audio_start:].float() - ref['audio'][:, :, audio_start:].float()).abs().max()),
                      same_decode_segmentation=True, NFE_per_chunk=meta['NFE'], model_evaluations=len(records) * meta['NFE'],
                      wall_sec=time.monotonic() - begin, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      host=__import__('socket').gethostname(), GPU=torch.cuda.get_device_name(device),
                      comparison_track='same_backbone_h3_engineering_diagnostic', generated_clock_only=True,
                      formal_memory_quality_claim=False, offline_audio_decode_only=True,
                      training_changed=False, default_policy_changed=False)
        temporary = out / (mode + '_RESULT.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(out / (mode + '_RESULT.json'))
        dist.barrier()
        if rank == 0:
            values = [json.loads((out / (m + '_RESULT.json')).read_text()) for m in ['carry', 'zero']]
            for a, b in zip(values[0]['records'], values[1]['records']):
                for key in ('row', 'initial_video_noise_sha256', 'initial_audio_noise_sha256', 'position_hashes_fp32', 'NFE'):
                    if a[key] != b[key]:
                        raise ValueError('Carry/zero inputs differ: ' + key)
            if values[0]['records'][0]['history_sha256'] != values[1]['records'][0]['history_sha256']:
                raise ValueError('First history differs')
            (out / 'CONTROL_RESULT.json').write_text(json.dumps(dict(same_first_prefix_and_noise=True,
                  window=args.window, modes=['carry', 'zero'], semantic_review_pending=True), indent=2) + '\n')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
