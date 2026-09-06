#!/usr/bin/env python3
"""Test SAM2 with physically prefix-only input directories against oracle masks.

This validates a mask producer on a recorded Dense trajectory, not an integrated
causal video generator. It does not infer VAE/SAM2 end-to-end online savings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_sam2_oracle_masks import EXPECTED_CHECKPOINT_SHA, select_mask, sha256


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('real CUDA producer gate required')
    root, out = Path(args.teacher_root).resolve(), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    producer = json.loads((root/'result.json').read_text())
    if producer['status'] != 'pass' or producer['manual_roi_used'] or producer['frames'] != 153:
        raise ValueError('audited automatic153-frame teacher required')
    start = time.perf_counter()
    checkpoint_sha = sha256(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA:
        raise ValueError('checkpoint source lock mismatch')
    checkpoint_verify_s = time.perf_counter()-start
    oracle = np.load(root/'pixel_masks.npz')['masks']
    initial = np.array(Image.open(root/'frames/00008.jpg').convert('RGB'), copy=True)
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    start = time.perf_counter()
    model = build_sam2_video_predictor('configs/sam2/sam2_hiera_l.yaml', args.checkpoint,
                                      device='cuda', apply_postprocessing=False)
    generator = SAM2AutomaticMaskGenerator(model, points_per_side=16, points_per_batch=32, min_mask_region_area=0)
    torch.cuda.synchronize()
    load_s = time.perf_counter()-start
    start = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        candidates = generator.generate(initial)
    torch.cuda.synchronize()
    initialize_s = time.perf_counter()-start
    chosen = select_mask(candidates, *initial.shape[:2])
    if chosen is None:
        raise ValueError('same automatic prefix did not initialize')
    records = []
    for length in (9, 69, 117):
        inputs = out/f'prefix{length}'
        inputs.mkdir()
        input_shas = []
        for i in range(length):
            source = root/f'frames/{i:05d}.jpg'
            (inputs/source.name).symlink_to(source)
            input_shas.append(sha256(source))
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            state = model.init_state(str(inputs), offload_video_to_cpu=True,
                                    offload_state_to_cpu=True, async_loading_frames=False)
            if state['num_frames'] != length:
                raise ValueError('predictor saw frames beyond the authorized prefix')
            model.add_new_mask(state, frame_idx=8, obj_id=1, mask=candidates[chosen]['segmentation'])
        torch.cuda.synchronize()
        state_init_s = time.perf_counter()-start
        masks = np.zeros_like(oracle[:length])
        seen = set()
        start = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for reverse in (True, False):
                for frame, _, logits in model.propagate_in_video(state, start_frame_idx=8,
                        max_frame_num_to_track=8 if reverse else length, reverse=reverse):
                    if frame < 0 or frame >= length:
                        raise ValueError('propagation escaped prefix')
                    masks[frame] = (logits[0] > 0).detach().cpu().numpy().squeeze()
                    seen.add(frame)
        torch.cuda.synchronize()
        propagation_s = time.perf_counter()-start
        difference = masks != oracle[:length]
        changed = np.flatnonzero(difference.reshape(length, -1).any(1)).tolist()
        record = {'prefix_pixel_frames': length, 'completed_latent_frames': (length+3)//4,
            'predictor_input_frames': int(state['num_frames']), 'observed_output_frames': len(seen),
            'all_prefix_masks_bitwise_equal': not changed and len(seen) == length,
            'changed_frames': changed, 'differing_pixel_fraction': float(difference.mean()),
            'state_init_s': state_init_s, 'propagation_s': propagation_s,
            'peak_allocated_gpu_bytes': torch.cuda.max_memory_allocated(), 'input_jpeg_sha256s': input_shas}
        records.append(record)
        np.savez_compressed(out/f'prefix{length}_masks.npz', masks=masks)
        print(json.dumps({k:v for k,v in record.items() if k != 'input_jpeg_sha256s'}), flush=True)
        del state
    result = {'status': 'pass' if all(r['all_prefix_masks_bitwise_equal'] for r in records) else 'negative',
        'scope': 'prefix_only_SAM2_producer_invariance_on_recorded_Dense_video', 'records': records,
        'checkpoint_sha256': checkpoint_sha, 'checkpoint_verify_s': checkpoint_verify_s,
        'oracle_masks_sha256': sha256(root/'pixel_masks.npz'), 'teacher_result_sha256': sha256(root/'result.json'),
        'model_load_s': load_s, 'automatic_initialization_s': initialize_s,
        'manual_roi_used': False, 'gpu': torch.cuda.get_device_name(),
        'VAE_prefix_equivalence_not_tested': True, 'integrated_causal_video_method': False,
        'prefix_only_input_not_just_future_output_ignored': True}
    with (out/'result.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
