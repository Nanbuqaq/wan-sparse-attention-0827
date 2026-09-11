#!/usr/bin/env python3
"""Actual source-mask routes, pre-return outputs and full video decode audit."""
import argparse
import hashlib
import json
from itertools import zip_longest
from pathlib import Path
import sys

import av
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.history_cache import tensor_sha256
from adapters.longlive_sparse.native_oracle_source_mask import fixed_mask_indices
from adapters.longlive_sparse.stable_source_mask_fill import stable_mask_indices


def main():
    p = argparse.ArgumentParser()
    for name in ('case', 'reference', 'mask', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--fill', choices=('uniform_midpoint', 'fixed_bit_reversal'), required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    load = lambda path: torch.load(path, map_location='cpu', weights_only=True)
    d = json.loads((args.case / 'summary.json').read_text())
    assert d['status'] == 'pass'
    current, old = load(args.case / 'latents.pt'), load(args.reference / 'latents.pt')
    assert torch.equal(current[:, :96], old[:, :96])
    assert tensor_sha256(current) == d['latent_sha256']
    payload = load(args.mask)
    assert tensor_sha256(current[:, payload['source_start']:payload['source_end']]) == payload['source_latent_sha256']
    if args.fill == 'uniform_midpoint':
        ids = fixed_mask_indices(payload['indices'], source_tokens=7040, budget=1760, mode='foreground')
    else:
        ids = stable_mask_indices(payload['indices'], source_tokens=7040, budget=1760)
    routes = load(args.case / 'causal_block_routes.pt')['records']
    assert len(routes) == 30 and sorted(r['layer'] for r in routes) == list(range(30))
    for route in routes:
        assert route['source_start'] == payload['source_start']
        actual = route['source_indices'].long()
        assert torch.equal(actual, ids[None].expand_as(actual))
    active = [r for r in d['causal_block_memory']['rows'] if r['active_source']]
    assert active and all(r['backend'] == 'native_FA2' and r['raw_unique_source_tokens_per_head'] == 1760 for r in active)
    frames, digest, samples = 0, hashlib.sha256(), []
    sample_frames = (188, 380, 396, 428, 460, 508)
    with av.open(str(args.case / 'video.mp4')) as ca, av.open(str(args.reference / 'video.mp4')) as cb:
        ca.streams.video[0].codec_context.thread_count = cb.streams.video[0].codec_context.thread_count = 2
        for i, (a, b) in enumerate(zip_longest(ca.decode(video=0), cb.decode(video=0))):
            assert a is not None and b is not None
            rgb = a.to_ndarray(format='rgb24')
            if i < 381:
                assert np.array_equal(rgb, b.to_ndarray(format='rgb24'))
            digest.update(memoryview(rgb)); frames += 1
            if i in sample_frames:
                tile = Image.fromarray(rgb).resize((320, 176))
                samples.append((i, tile))
    assert frames == 509
    contact = Image.new('RGB', (1920, 200), 'white')
    draw = ImageDraw.Draw(contact)
    for col, (frame, tile) in enumerate(samples):
        contact.paste(tile, (320 * col, 24)); draw.text((320 * col + 4, 4), f'{args.case.name} frame {frame}', fill='black')
    contact.save(args.output / 'contact.jpg', quality=95)
    report = dict(status='pass', actual_pre96_latents_and_pre381_decoded_RGB_exact=True,
        actual_source_hash_verified=True, actual_30_routes_exact=True, selected_tokens=1760,
        mask_tokens=payload['indices'].numel(), frames=frames, decoded_sha256=digest.hexdigest(),
        fill=args.fill, generation_s=d['native_DiT_s'], delivery_s=d['video_pipeline']['complete_s'],
        quality='pending qualitative review; technical pass is not identity preservation',
        precomputed_masks_excluded_from_online_speed_claim=True)
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
