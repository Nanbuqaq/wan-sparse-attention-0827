#!/usr/bin/env python3
"""SAM2 dependency/initialization cost probe on a completed-video prefix.

An offline prefix probe, NOT a validated online TetherMem method. No future
decoded image is supplied to SAM2, and no manual ROI or box selects the mask.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import time
import av
import numpy as np
from PIL import Image
import torch


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--video',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--completed-frames',type=int,default=9)
    parser.add_argument('--points-per-side',type=int,default=8)
    args=parser.parse_args()
    if args.completed_frames<1 or args.points_per_side<1 or not torch.cuda.is_available():
        raise ValueError('positive completed prefix and real CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter()
    with av.open(args.video) as video:
        prefix=[frame.to_ndarray(format='rgb24') for frame in itertools.islice(video.decode(video=0),args.completed_frames)]
    if len(prefix)!=args.completed_frames:
        raise ValueError('video shorter than declared completed prefix')
    decode_s=time.perf_counter()-begin
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    model=build_sam2('configs/sam2/sam2_hiera_l.yaml',args.checkpoint,
                     device='cuda',apply_postprocessing=False)
    generator=SAM2AutomaticMaskGenerator(model,points_per_side=args.points_per_side,points_per_batch=32,
                                        min_mask_region_area=0)
    torch.cuda.synchronize()
    load_s=time.perf_counter()-start
    start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        masks=generator.generate(prefix[-1])
    torch.cuda.synchronize()
    generation_s=time.perf_counter()-start
    height,width=prefix[-1].shape[:2]
    records=[]
    for index,mask in enumerate(masks):
        area=mask['area']/(height*width)
        x,y,w,h=mask['bbox']
        eligible=(.02<=area<=.8 and w<.95*width and h<.95*height
                  and bool(mask['segmentation'][height//2,width//2]))
        records.append({'index':index,'area_fraction':area,'bbox':mask['bbox'],
            'predicted_iou':mask['predicted_iou'],'stability_score':mask['stability_score'],
            'eligible_central_foreground':eligible})
    eligible=[row for row in records if row['eligible_central_foreground']]
    selected=max(eligible,key=lambda row:(row['area_fraction'],row['predicted_iou'],-row['index'])) if eligible else None
    Image.fromarray(prefix[-1]).save(root/'completed_prefix_last_frame.png')
    if masks:
        np.savez_compressed(root/'automatic_masks.npz',masks=np.stack([mask['segmentation'] for mask in masks]))
    if selected:
        mask=masks[selected['index']]['segmentation']
        overlay=prefix[-1].astype(float)
        overlay[mask]=overlay[mask]*.55+np.array([0,220,80])*.45
        Image.fromarray(overlay.astype(np.uint8)).save(root/'automatic_candidate_overlay.png')
    result={'status':'pass' if selected else 'negative',
        'segmentation_runtime_status':'pass' if masks else 'negative',
        'automatic_initialization_status':'candidate_requires_review' if selected else 'negative_no_eligible_mask',
        'scope':'offline_completed_prefix_initialization_cost_probe',
        'source_video':args.video,'prefix_images_decoded':len(prefix),'sam2_input_frame_index':len(prefix)-1,
        'sam2_future_image_inputs':0,'manual_roi_used':False,'semantic_subject_ground_truth':False,
        'online_runtime_validated':False,'mask_selection_rule':'largest central non-full-frame mask, area 2-80 percent',
        'selected_candidate':selected,'automatic_candidates':records,'cpu_prefix_decode_s':decode_s,
        'model_load_s':load_s,'first_mask_generation_wall_s':generation_s,
        'model_plus_first_mask_wall_s':load_s+generation_s,'gpu':torch.cuda.get_device_name(),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'points_per_side':args.points_per_side,'points_per_batch':32,
        'connected_component_postprocessing':False,
        'measurement_limit':'first-call startup included; model load not amortized; no propagation or VAE cost measured'}
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='automatic_candidates'},indent=2))


if __name__=='__main__':
    main()
