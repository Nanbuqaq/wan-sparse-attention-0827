#!/usr/bin/env python3
"""Score a matched group from canonical raw VAE pixels, never lossy MP4.

Identical latent values reuse one render/metric result, not independent samples.
Generation latency remains in the original runs; this is separate offline eval.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.offline_eval import output_error_metrics
from adapters.longlive_sparse.video_decode import decode_latents_chunked_exact
from scripts.evaluate_videos import psnr, ssim, lpips_distances
from scripts.build_video_review_storyboards import storyboard, quarter_sample_indices


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def metric_pixels(raw):
    """Pixel artifacts are uint8; shared quality functions require float [0,1]."""
    if raw.dtype != np.uint8:
        raise ValueError('canonical artifact input must be uint8 RGB')
    return raw.astype(np.float32)/255.0


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--states', required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--latent-frames', type=int, required=True)
    p.add_argument('--linear-weights', required=True)
    p.add_argument('--trunk-weights', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--expected', help='frozen batch manifest; required by formal batch driver')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('real GPU VAE/LPIPS evaluation required')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    all_cases = json.loads(Path(args.states).read_text())['cases']
    cases = [c for c in all_cases if c['prompt_id'] == args.prompt and c['seed'] == args.seed
             and c['latent_frames'] == args.latent_frames]
    expected_group = {}
    if args.expected:
        expected = json.loads(Path(args.expected).read_text())
        expected_group = {c['id']: c for c in expected['cases'] if c['prompt_id'] == args.prompt
                          and c['seed'] == args.seed and c['latent_frames'] == args.latent_frames}
        if set(expected_group) != {c['id'] for c in cases} or len(cases) != len(expected_group):
            raise ValueError('canonical evaluation omitted/duplicated a frozen case')
        if any(c['case_key'] != expected_group[c['id']]['case_key'] for c in cases):
            raise ValueError('canonical input changed a frozen case identity')
    dense = [c for c in cases if c['method'] == 'rag_dense' and c['status'] == 'pass']
    if len(dense) != 1:
        raise ValueError('exactly one matched Dense reference required')
    reference = dense[0]
    for case in cases:
        if case['status'] != 'pass':
            continue
        for key in ('seed', 'prompt_sha256', 'latent_frames', 'rope_policy'):
            if case['case_key'][key] != reference['case_key'][key]:
                raise ValueError('unmatched canonical quality group')
        if case['initial_noise_sha256'] != reference['initial_noise_sha256']:
            raise ValueError('initial noise differs across paired methods')
    sys.path.insert(0, os.environ['LONGLIVE_BASE_SOURCE'])
    from utils.wan_wrapper import WanVAEWrapper, _wan_model_dir
    vae_weight = Path(_wan_model_dir())/'Wan2.1_VAE.pth'
    weight_sha = sha(vae_weight)
    start = time.perf_counter()
    vae = WanVAEWrapper().eval().requires_grad_(False).to(device='cuda', dtype=torch.bfloat16)
    torch.cuda.synchronize()
    model_load_s = time.perf_counter()-start
    metric = json.loads((ROOT/'configs/quality/lpips_alex_v0p1.json').read_text())
    renders, quality_cache, rows = {}, {}, []
    ref_latent = torch.load(Path(reference['video']).parent/'latents.pt', map_location='cpu', weights_only=True)
    ref_hash = tensor_sha256(ref_latent)

    def render(latent, digest):
        started = time.perf_counter()
        video = decode_latents_chunked_exact(vae, latent.cuda(), chunk_size=120)
        rgb = ((video*.5+.5).clamp(0, 1)*255).to(torch.uint8).permute(0, 1, 3, 4, 2).contiguous()[0]
        torch.cuda.synchronize()
        render_s = time.perf_counter()-started
        if len(rgb) != 4*args.latent_frames-3:
            raise ValueError('canonical VAE frame count mismatch')
        rgb_sha = tensor_sha256(rgb)
        directory = out/digest[:16]
        directory.mkdir()
        quarters = quarter_sample_indices(len(rgb), 16)
        overview = np.linspace(0, len(rgb)-1, 16).round().astype(int)
        chosen = set(overview.tolist()) | {int(i) for quarter in quarters for i in quarter}
        thumbnails = [None]*len(rgb)
        for index in chosen:
            thumbnails[index] = np.asarray(Image.fromarray(rgb[index].numpy()).resize((208, 120)))
        storyboard(thumbnails, overview, directory/'overview.png')
        for number, indices in enumerate(quarters, 1):
            storyboard(thumbnails, indices, directory/f'quarter{number}.png')
        detail_frames = [0, len(rgb)//4-1, len(rgb)//2-1, 3*len(rgb)//4-1, len(rgb)-1]
        for index in detail_frames:
            Image.fromarray(rgb[index].numpy()).save(directory/f'detail_{index:04d}.png')
        meta = {'latent_sha256': digest, 'raw_rgb_sha256': rgb_sha, 'frames': len(rgb), 'render_s': render_s,
            'render_directory': str(directory.resolve()), 'detail_frame_indices': detail_frames,
            'vae_checkpoint_sha256': weight_sha, 'source': 'cache_continuous_raw_VAE_u8_no_codec'}
        (directory/'render.json').write_text(json.dumps(meta, indent=2)+'\n')
        renders[digest] = meta
        return metric_pixels(rgb.numpy())

    ref_rgb = render(ref_latent, ref_hash)
    for case in sorted(cases, key=lambda c: (c['id'] != reference['id'], c['id'])):
        if case['status'] != 'pass':
            rows.append({'case_id': case['id'], 'status': case['status'], 'failure_reason': case.get('failure_reason')})
            continue
        path = Path(case['video']).parent/'latents.pt'
        latent = torch.load(path, map_location='cpu', weights_only=True)
        digest = tensor_sha256(latent)
        if digest == ref_hash:
            values = {'lpips_mean': 0., 'late_quarter_lpips_mean': 0., 'psnr_mean': None, 'ssim_mean': 1.,
                'raw_rgb_exact_dense': True, 'latent_error': output_error_metrics(ref_latent, latent),
                'quarter_lpips': [0., 0., 0., 0.]}
        elif digest in quality_cache:
            values = quality_cache[digest]
        else:
            pixels = render(latent, digest)
            distances, error, provenance = lpips_distances(ref_rgb, pixels, weights_path=Path(args.linear_weights),
                expected_sha256=metric['linear_weights']['sha256'], expected_version=metric['package_versions']['lpips'],
                trunk_weights_path=Path(args.trunk_weights), expected_trunk_sha256=metric['trunk_weights']['sha256'],
                expected_torch_version=metric['package_versions']['torch'], expected_torchvision_version=metric['package_versions']['torchvision'])
            if error or distances is None:
                raise RuntimeError(f'locked raw-pixel LPIPS unavailable: {error}')
            quarter_bounds = np.linspace(0, len(pixels), 5).round().astype(int)
            values = {'lpips_mean': float(np.mean(distances)), 'late_quarter_lpips_mean': float(np.mean(distances[3*len(pixels)//4:])),
                'psnr_mean': float(np.mean([psnr(a, b) for a, b in zip(ref_rgb, pixels)])),
                'ssim_mean': float(np.mean([ssim(a, b) for a, b in zip(ref_rgb, pixels)])),
                'quarter_lpips': [float(np.mean(distances[a:b])) for a, b in zip(quarter_bounds[:-1], quarter_bounds[1:])],
                'lpips_per_frame': list(map(float, distances)), 'latent_error': output_error_metrics(ref_latent, latent),
                'raw_rgb_exact_dense': False, 'lpips_provenance': provenance}
            quality_cache[digest] = values
            del pixels
        rows.append({'case_id': case['id'], 'method': case['method'], 'status': 'pass', 'latent_sha256': digest,
            'formal_config_id': expected_group.get(case['id'], {}).get('formal_config_id'),
            'latent_artifact_sha256': sha(path), 'canonical_render': renders[digest], 'system': case['case_key']['system'],
            **values, 'absolute_semantic_quality_proven': False})
        print(json.dumps({'case': case['id'], 'lpips': values['lpips_mean'], 'raw_render': digest[:16]}), flush=True)
    result = {'status': 'pass', 'scope': 'canonical_raw_pixel_quality_not_lossy_preview', 'prompt': args.prompt, 'seed': args.seed,
        'latent_frames': args.latent_frames, 'reference_case': reference['id'], 'rows': rows,
        'unique_latent_renders': len(renders), 'identical_latents_not_independent_quality_samples': True,
        'gpu': torch.cuda.get_device_name(), 'model_load_s': model_load_s, 'vae_checkpoint_sha256': weight_sha,
        'decode_policy': 'BF16_cache_continuous_chunk120_raw_u8', 'generation_timing_not_replaced_by_eval_time': True,
        'MP4_ignored_for_quality': True, 'raw_pixel_arrays_reproducible_from_pinned_latents_weights_and_policy': True,
        'metric_input_protocol': 'canonical_uint8_RGB_cast_float32_div255_before_LPIPS_PSNR_SSIM',
        'metric_input_protocol_version': 2}
    result['provenance'] = {'input_states_sha256': sha(args.states),
        'expected_manifest_sha256': sha(args.expected) if args.expected else None,
        'evaluation_source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'evaluation_script_sha256': sha(__file__), 'metric_protocol_sha256': sha(ROOT/'configs/quality/lpips_alex_v0p1.json'),
        'generation_commits': sorted({c['case_key']['commit'] for c in cases})}
    (out/'quality.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
