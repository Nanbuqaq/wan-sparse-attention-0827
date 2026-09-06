#!/usr/bin/env python3
"""Audit raw RGB equality and match diagnostic latent/route trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.history_cache import tensor_sha256


def ordered_routes(case):
    data = json.loads(Path(case['stats']).read_text())
    return [(r['layer_id'], r['current_start'], r['denoising_pass'], r['route_plan_sha256']) for r in data['call_records']]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--diagnostic-root', required=True)
    p.add_argument('--original-root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    root = Path(args.diagnostic_root).resolve()
    cases = json.loads((root/'recovered_states.json').read_text())['cases']
    original = json.loads((Path(args.original_root)/'recovered_states.json').read_text())['cases']
    groups = []
    for lane in (0, 1):
        pair = [c for c in cases if c['lane'] == lane]
        if len(pair) != 2 or any(c['status'] != 'pass' for c in pair):
            raise ValueError('complete diagnostic pair required')
        rgb, latents, routes = [], [], []
        for case in pair:
            path = Path(case['video']).parent
            pixels = torch.load(path/'raw_rgb_frames.pt', map_location='cpu', weights_only=True)
            if tensor_sha256(pixels) != case['raw_video_rgb_sha256']:
                raise ValueError('raw pixel dump digest mismatch')
            latent = torch.load(path/'latents.pt', map_location='cpu', weights_only=True)
            reference = next(c for c in original if c['method'] == case['method'] and c['prompt_id'] == case['prompt_id']
                and c['seed'] == case['seed'] and c['case_key']['system']['gpu_union_cache'] == case['case_key']['system']['gpu_union_cache'])
            old_latent = torch.load(Path(reference['video']).parent/'latents.pt', map_location='cpu', weights_only=True)
            if not torch.equal(latent, old_latent) or ordered_routes(case) != ordered_routes(reference):
                raise ValueError('instrumented trajectory does not reproduce the specific original run')
            rgb.append(pixels)
            latents.append(latent)
            routes.append(ordered_routes(case))
        same_raw = torch.equal(*rgb)
        same_float = pair[0]['raw_video_float_sha256'] == pair[1]['raw_video_float_sha256']
        same_latent = torch.equal(*latents)
        same_routes = routes[0] == routes[1]
        if not all((same_raw, same_float, same_latent, same_routes)):
            raise ValueError('preencode system equivalence failed')
        groups.append({'lane': lane, 'method': pair[0]['method'], 'prompt': pair[0]['prompt_id'],
            'original_93db4ab_trajectories_reproduced': True, 'all_latents_equal': same_latent,
            'ordered_routes_equal': same_routes, 'raw_VAE_float_hashes_equal': same_float,
            'raw_RGB_tensors_and_hashes_equal': same_raw, 'RGB_strides': pair[0]['raw_video_rgb_strides'],
            'MP4_file_hashes_equal': pair[0]['video_sha256'] == pair[1]['video_sha256'],
            'raw_RGB_sha256': pair[0]['raw_video_rgb_sha256'], 'raw_float_sha256': pair[0]['raw_video_float_sha256'],
            'case_ids': [c['id'] for c in pair]})
    result = {'status': 'pass', 'cases': 4, 'missing': 0, 'groups': groups,
        'scope': 'exact_original_trajectory_and_preencode_equivalence_not_new_quality_or_speed_samples',
        'localization': 'production_encoding_path_after_identical_float_and_uint8_RGB',
        'specific_codec_root_cause': 'not proven; isolated identical contiguous RGB encoding was deterministic',
        'decision': 'use raw_RGB_and_latents_for_model_quality; MP4_SHA_is_artifact_integrity_not_semantic_equivalence',
        'speed_claim': False, 'hierarchy_promoted': False,
        'states_sha256': hashlib.sha256((root/'recovered_states.json').read_bytes()).hexdigest()}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
