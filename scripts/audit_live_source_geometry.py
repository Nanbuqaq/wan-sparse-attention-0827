#!/usr/bin/env python3
"""Independent CPU checks for the registered live source geometry integration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import av
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.history_cache import tensor_sha256


def main():
    p = argparse.ArgumentParser()
    for name in ('case', 'reference', 'raw-reference', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--raw-reference-subset',action='store_true',help='external raw replay may cover only selected archives; all selected sources remain mandatory')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    read = lambda root: json.loads((root / 'summary.json').read_text())
    load = lambda root, name: torch.load(root / name, weights_only=True, map_location='cpu')
    d, ref = read(args.case), read(args.reference)
    assert d['status'] == ref['status'] == 'pass'
    latents = load(args.case, 'latents.pt')
    assert torch.equal(latents, load(args.reference, 'latents.pt'))
    assert tensor_sha256(latents) == d['latent_sha256']
    assert d['pixels']['raw_RGB_sha256'] == ref['pixels']['raw_RGB_sha256']
    raw = load(args.case, 'source_raw_rgb.pt')
    old_raw = load(args.raw_reference, 'source_raw_rgb.pt')
    assert old_raw['complete'] and old_raw['pixels_before_lossy_codec'], 'unqualified external raw reference'
    assert raw['complete'] and len(raw['records']) == 3
    old = {r['archive_version']: r for r in old_raw['records']}
    geometry = load(args.case, 'source_geometry.pt')['results']
    sources = {};matched_raw_versions=set()
    for r in raw['records']:
        version = r['archive_version']
        assert r['source_latent_sha256'] == tensor_sha256(latents[:, r['source_start']:r['source_end']])
        assert r['raw_pixel_bytes_sha256'] == hashlib.sha256(memoryview(r['pixels'].numpy())).hexdigest()
        # Only pre-return source windows are invariant under partial history selection.
        if r['source_end'] <= 96:
            if version in old:
                assert torch.equal(r['pixels'], old[version]['pixels'])
                matched_raw_versions.add(version)
            else:
                assert args.raw_reference_subset, 'external raw reference missing an archive'
        if geometry[version]['status'] == 'mask_ready':
            assert geometry[version]['source_raw_pixel_sha256'] == r['raw_pixel_bytes_sha256']
            assert geometry[version]['source_latent_sha256'] == r['source_latent_sha256']
        sources[version] = r
    selected_versions={r['archive_version'] for r in d['causal_block_memory']['geometry_sources']}
    assert selected_versions and selected_versions<=matched_raw_versions
    del old_raw, old
    routes = load(args.case, 'causal_block_routes.pt')['records']
    reference_routes = load(args.reference, 'causal_block_routes.pt')['records']
    assert len(routes) == len(reference_routes) == 30
    for a, b in zip(routes, reference_routes):
        assert a.keys() == b.keys()
        for key in a:
            assert torch.equal(a[key], b[key]) if isinstance(a[key], torch.Tensor) else a[key] == b[key]
    assert d['causal_block_memory']['geometry_used_layers'] == list(range(30))
    decoded = {}
    for label, root in (('live', args.case), ('reference', args.reference)):
        digest, count = hashlib.sha256(), 0
        with av.open(str(root / 'video.mp4')) as container:
            container.streams.video[0].codec_context.thread_count = 2
            for frame in container.decode(video=0):
                digest.update(memoryview(frame.to_ndarray(format='rgb24')))
                count += 1
        decoded[label] = dict(frames=count, sha256=digest.hexdigest())
    assert decoded['live'] == decoded['reference'] and decoded['live']['frames'] == 509
    report = dict(status='pass', actual_full_latents_raw_RGB_and_decoded_RGB_equal=True,
        all_30_actual_routes_equal=True, raw_source_ownership_verified=True, decoded=decoded,
        geometry=d['source_geometry'], geometry_model=d['source_geometry_model'],
        witness=d['source_pixel_witness'],
        external_raw_reference_versions_verified=sorted(matched_raw_versions),
        external_raw_reference_scope=('selected sources; other source windows only internally hash-checked'
            if args.raw_reference_subset else 'all registered source windows'),
        GPU1_peak_bytes=d['pipeline_VAE_GPU_peak_allocated_bytes'],
        limitations=['one development case, not generalization or a latency win',
            'geometry H2D and kernel attribution remain unmeasured'])
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status='pass', frames=509, routes=30)))


if __name__ == '__main__':
    main()
