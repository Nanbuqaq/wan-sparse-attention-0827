#!/usr/bin/env python3
"""Real VAE continuous stream/completion-thread gate on saved completed latents."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.streaming_vae import StreamingVAEDecoder
from adapters.longlive_sparse.video_decode import decode_latents_chunked_exact
from adapters.longlive_sparse.history_cache import tensor_sha256


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--latent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('preserve previous gate')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('real CUDA required')
    sys.path.insert(0, os.environ['LONGLIVE_BASE_SOURCE'])
    from utils.wan_wrapper import WanVAEWrapper
    from utils.misc import set_seed
    set_seed(20260904)
    latent = torch.load(args.latent, map_location='cpu', weights_only=True).cuda()
    if latent.shape[0] != 1 or latent.shape[1] % 3:
        raise ValueError('batch-one, three-latent-aligned input required')
    start = time.perf_counter()
    vae = WanVAEWrapper().eval().requires_grad_(False).to(device='cuda', dtype=torch.bfloat16)
    torch.cuda.synchronize()
    load_s = time.perf_counter()-start
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    reference = decode_latents_chunked_exact(vae, latent, chunk_size=120)
    full_wall = time.perf_counter()-start
    full_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    decoder = StreamingVAEDecoder(vae, device=latent.device, dtype=latent.dtype)
    for offset in range(0, latent.shape[1], 3):
        decoder.submit(latent[:, offset:offset+3], start_latent=offset)
    output, metrics = decoder.finish()
    matched = bool(torch.equal(reference, output))
    result = dict(status='pass' if matched else 'negative', scope='saved_completed_latent_stream_decode_not_generation_overlap',
        latent_file_sha256=hashlib.sha256(args.latent.read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('scripts/gate_streaming_vae.py', 'adapters/longlive_sparse/streaming_vae.py')},
        gpu=torch.cuda.get_device_name(), model_load_s=load_s, batch_decode_wall_s=full_wall,
        batch_peak_GPU_bytes=full_peak, stream_peak_GPU_bytes=torch.cuda.max_memory_allocated(),
        raw_pixels_bitwise_equal=matched, max_abs=float((reference-output).abs().max()),
        raw_reference_sha256=tensor_sha256(reference), raw_stream_sha256=tensor_sha256(output), stream=metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'stream'}), flush=True)
    print(json.dumps({k: v for k, v in metrics.items() if k != 'records'}), flush=True)
    if not matched:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
