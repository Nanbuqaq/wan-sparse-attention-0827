#!/usr/bin/env python3
"""Compare public and compact SAM2 outputs on actual source2/3 image embeddings."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.longlive_sparse.cached_source_geometry import CachedSourceGeometry
from adapters.longlive_sparse.compact_source_mask import predict_selected_mask, verify_predictor_source, PREDICTOR_SHA
from adapters.longlive_sparse.native_raw_source_input import load_raw_source_window


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    for name in ('case', 'checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    summary = json.loads((args.case / 'summary.json').read_text())
    geometry = torch.load(args.case / 'source_geometry.pt', map_location='cpu', weights_only=True)['results']
    model = CachedSourceGeometry(args.checkpoint)
    verify_predictor_source(model.predictor)
    cpu_rng, gpu_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    rows = []
    with torch.cuda.device(model.device), torch.cuda.stream(model.stream), torch.autocast('cuda', dtype=torch.bfloat16):
        for version in (2, 3):
            window = load_raw_source_window(args.case / 'source_raw_rgb.pt', version, summary)
            box = geometry[version]['bbox']
            for frame, pixels in enumerate(window['pixels']):
                model.predictor.set_image(pixels.numpy())
                model.stream.synchronize()
                # Alternate order to avoid a fixed warm-cache timing advantage.
                calls = ('public', 'compact') if frame % 2 == 0 else ('compact', 'public')
                outputs, times = {}, {}
                for kind in calls:
                    began = time.perf_counter()
                    if kind == 'public':
                        outputs[kind] = model.predictor.predict(box=np.asarray(box, dtype=np.float32), multimask_output=True)
                    else:
                        outputs[kind] = predict_selected_mask(model.predictor, box)
                    model.stream.synchronize()
                    times[kind] = time.perf_counter() - began
                masks, scores, logits = outputs['public']
                selected, compact_scores, chosen = outputs['compact']
                assert chosen == int(np.argmax(scores))
                assert np.array_equal(scores, compact_scores)
                assert np.array_equal(selected, masks[chosen].astype(np.bool_))
                rows.append(dict(version=version, pixel_frame=window['pixel_start'] + frame,
                    exact=True, chosen=chosen, scores=scores.tolist(),
                    public_return_bytes=masks.nbytes + scores.nbytes + logits.nbytes,
                    compact_return_bytes=selected.nbytes + compact_scores.nbytes,
                    public_wall_s=times['public'], compact_wall_s=times['compact']))
    model.predictor.reset_predictor()
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state())
    result = dict(status='pass', frames=len(rows), rows=rows, predictor_source_sha256=PREDICTOR_SHA,
        actual_GPU=torch.cuda.get_device_name(), CPU_and_GPU_RNG_unchanged=True,
        public_return_bytes=sum(r['public_return_bytes'] for r in rows),
        compact_return_bytes=sum(r['compact_return_bytes'] for r in rows),
        public_wall_s=sum(r['public_wall_s'] for r in rows),
        compact_wall_s=sum(r['compact_wall_s'] for r in rows),
        limitations=['shared image embeddings; whole online extraction still requires an integration gate',
            'returned tensor bytes are not a PCIe hardware counter or complete SAM transfer cost'])
    (args.output / 'gate.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
