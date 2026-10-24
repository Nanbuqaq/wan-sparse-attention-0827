"""Stage1 step8 causal health worker.

Runs the 7+5+5+5 Stage1 causal KV contract and writes per-chunk latent and
timing evidence. It never invokes an optimizer or training update.
"""
import argparse, datetime, json, os, sys, time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    args = ap.parse_args()
    request = json.loads(Path(args.request).read_text())
    source = Path(request['stage1_source_dir'])
    sys.path.insert(0, str(source))
    from train_h3_worldattention_dmd_fsdp2 import CHUNKS, build_student, generation_step
    from train_h3_worldattention_ode_fsdp2 import chunk_contract
    from h3_worldattention_causal import VideoKVBank

    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK']); world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local); device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(hours=2), device_id=device)
    mesh = init_device_mesh('cuda', (world,)); out = Path(os.environ['INFER_OUTPUT_DIR']); out.mkdir(parents=True, exist_ok=True)
    try:
        if world != 8: raise ValueError('Stage1 health requires eight ranks')
        prompt = json.loads(Path(request['prompt_pool']).read_text().splitlines()[0])
        emb = torch.load(prompt['cached_embedding_path_candidate'], map_location='cpu', weights_only=True)
        video = torch.zeros(24, 22, 48, 84, dtype=torch.bfloat16)
        audio = torch.zeros(2, 32, 122, dtype=torch.bfloat16)
        model, processors = build_student(request['base_source'], request['export'], mesh, device, 'student', True, 8192, 4096, rank)
        bank = VideoKVBank(local_frames=12, sink_frames=3)
        generator = torch.Generator(device=device).manual_seed(request['seed'])
        records = []
        for chunk, (start, end) in enumerate(CHUNKS):
            audio_end = 122 if end == 22 else round((5 + 17 * ((end - 2) // 5)) * 40 / 24)
            batch = chunk_contract(video, audio, emb['prompt_embeds'], emb['text_token_tags'], generator, device, start, end, audio_end)
            tick = time.monotonic()
            generated, generated_audio, initial = generation_step(model, processors, batch, chunk, start, end, bank, generator, request['nfe'], device, False)
            torch.cuda.synchronize(); elapsed = time.monotonic() - tick
            finite = bool(torch.isfinite(generated).all() and torch.isfinite(generated_audio).all())
            record = dict(chunk=chunk, start_latent=start, end_latent=end, latent_mean=float(generated.float().abs().mean()),
                          latent_std=float(generated.float().std()), latent_max=float(generated.float().abs().max()), finite=finite,
                          wall_seconds=elapsed, kv_bytes=bank.bytes)
            records.append(record)
            torch.save(dict(video=generated.cpu(), audio=generated_audio.cpu(), initial_video=initial[0].cpu(), initial_audio=initial[1].cpu(), record=record), out / f'chunk{chunk:02d}.pt')
            if not finite: raise RuntimeError('Nonfinite Stage1 output')
        if rank == 0:
            (out / 'RESULT.json').write_text(json.dumps(dict(status='stage1_step8_four_chunk_complete', checkpoint=request['export'], nfe=request['nfe'], chunks=records, history_contract='causal_local12_sink3_readonly_video_KV'), indent=2) + '\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__': main()
