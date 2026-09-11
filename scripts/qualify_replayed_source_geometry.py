#!/usr/bin/env python3
"""Same-platform raw source geometry gate and independent offline mask payloads."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.cached_source_geometry import CachedSourceGeometry
from adapters.longlive_sparse.compact_source_mask import verify_predictor_source
from adapters.longlive_sparse.native_raw_source_input import load_raw_source_window
from scripts.probe_native_source_foreground import source_token_masks


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for name in ('replay','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    summary=json.loads((args.replay/'summary.json').read_text())
    if summary['status']!='pass' or not summary['original_raw_RGB_exact']:
        raise ValueError('only exact original raw RGB replay may seed platform geometry')
    window=load_raw_source_window(args.replay/'source_raw_rgb.pt',2,summary)
    model=CachedSourceGeometry(args.checkpoint);cpu_rng=torch.random.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
    public=model.extract(window);verify_predictor_source(model.predictor);model.compact_return=True
    compact=model.extract(window)
    for key in ('pixel_masks','token_masks','indices'):
        if not torch.equal(public[key],compact[key]):raise RuntimeError('same-platform compact return changed masks')
    if public['records']!=compact['records']:raise RuntimeError('same-platform scores or choices differ')
    if not torch.equal(cpu_rng,torch.random.get_rng_state()) or not torch.equal(gpu_rng,torch.cuda.get_rng_state()):
        raise RuntimeError('geometry changed generator RNG state')
    masks=public['pixel_masks'].numpy();rows=[]
    for name,stride in (('all32',1),('holdfirst',32)):
        used=masks[(np.arange(len(masks))//stride)*stride]
        tokens=source_token_masks(used,window['source_start']);indices=torch.from_numpy(np.flatnonzero(tokens.flatten()).astype(np.int64))
        out=args.output/name;out.mkdir()
        payload=dict(schema='native_past_source_mask_v1',source_start=window['source_start'],source_end=window['source_end'],
            source_latent_sha256=window['source_latent_sha256'],pixel_input_kind='raw_stream_before_codec',
            source_pixel_sha256=window['raw_pixel_bytes_sha256'],indices=indices,token_grid=[22,40],
            scope='offline platform-matched mask from exact replayed original raw pixels; no online cost claim',
            manual_bbox=False,prompt_bbox=public['bbox'],mask_stride=stride)
        with (out/'source_mask_indices.pt').open('xb') as f:torch.save(payload,f)
        rows.append(dict(method=name,source_tokens=indices.numel(),fits1760=indices.numel()<=1760,
            box=public['bbox'],payload=str(out/'source_mask_indices.pt')))
    with (args.output/'geometry.pt').open('xb') as f:torch.save(public,f)
    result=dict(status='pass',same_platform_public_compact_masks_scores_choices_exact=True,
        CPU_GPU_RNG_unchanged=True,GPU=torch.cuda.get_device_name(),rows=rows,
        source_latent_sha256=window['source_latent_sha256'],raw_source_sha256=window['raw_pixel_bytes_sha256'],
        no_video_or_quality_claim=True)
    (args.output/'gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
