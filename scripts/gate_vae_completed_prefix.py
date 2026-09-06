#!/usr/bin/env python3
"""VAE prefix/incremental equivalence and external decode cost on saved latents."""
from __future__ import annotations
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
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.video_decode import decode_latents_chunked_exact


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', required=True)
    p.add_argument('--kind', choices=('motion', 'state'), required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('real CUDA required')
    root = Path(args.workspace).resolve()
    states = json.loads((root/'results/videos/aligned_final_cf5c25f_local/states.json').read_text())['cases']
    case = next(c for c in states if c['method'] == 'rag_dense' and c['prompt_id'] == f'calibration_{args.kind}')
    latent_path = Path(case['video']).parent/'latents.pt'
    latent = torch.load(latent_path, map_location='cpu', weights_only=True)
    if latent.shape != (1, 39, 16, 60, 104):
        raise ValueError('matching39-latent Dense reference required')
    sys.path.insert(0, os.environ['LONGLIVE_BASE_SOURCE'])
    from utils.wan_wrapper import WanVAEWrapper
    started = time.perf_counter()
    vae = WanVAEWrapper().eval().requires_grad_(False).to(device='cuda', dtype=torch.bfloat16)
    torch.cuda.synchronize()
    load_s = time.perf_counter()-started
    latent = latent.cuda()
    def decode(value, chunks):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        result = decode_latents_chunked_exact(vae, value, chunk_size=chunks)
        torch.cuda.synchronize()
        return result, time.perf_counter()-t, torch.cuda.max_memory_allocated()
    full, full_s, full_peak = decode(latent, 39)
    rows = []
    for length in (3, 18, 30):
        prefix, elapsed, peak = decode(latent[:, :length], length)
        target = full[:, :4*length-3]
        rows.append({'completed_latents': length, 'pixels': len(prefix[0]),
            'raw_pixels_bitwise_equal': torch.equal(prefix, target),
            'max_abs': float((prefix-target).abs().max()), 'decode_with_D2H_s': elapsed,
            'peak_allocated_gpu_bytes': peak})
        del prefix
    incremental, incremental_s, incremental_peak = decode(latent, 3)
    matched = bool(torch.equal(incremental, full))
    result = {'status': 'pass' if matched and all(r['raw_pixels_bitwise_equal'] for r in rows) else 'negative',
        'scope': 'completed-prefix_and_continuous_three-latent_incremental_VAE_on_saved_Dense_latents',
        'kind': args.kind, 'case_id': case['id'], 'generation_commit': case['case_key']['commit'],
        'latent_artifact_sha256': hashlib.sha256(latent_path.read_bytes()).hexdigest(),
        'model_load_s': load_s, 'full39_decode_with_D2H_s': full_s,
        'incremental3_decode_with_D2H_s': incremental_s,
        'full_peak_allocated_gpu_bytes': full_peak, 'incremental_peak_allocated_gpu_bytes': incremental_peak,
        'full_raw_pixel_sha256': tensor_sha256(full), 'incremental_raw_pixels_bitwise_equal': matched,
        'incremental_max_abs': float((incremental-full).abs().max()), 'prefix_checks': rows,
        'gpu': torch.cuda.get_device_name(), 'recorded_trajectory_only_not_online_generation': True,
        'SAM2_and_video_encoding_not_included': True}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
