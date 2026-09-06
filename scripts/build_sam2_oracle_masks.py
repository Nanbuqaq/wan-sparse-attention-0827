#!/usr/bin/env python3
"""Automatic completed-prefix initialization and full-video offline SAM2 teacher.

No manual box, no online speed/quality claim. Backward prefix propagation reads
only the already-completed initial chunk; the full forward run is offline.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import av
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.causal_roles import align_pixel_patch_masks_to_latent

EXPECTED_CHECKPOINT_SHA='7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def select_mask(masks,height,width):
    eligible=[]
    for i,m in enumerate(masks):
        x,y,w,h=m['bbox'];fraction=m['area']/(height*width)
        if .02<=fraction<=.8 and w<.95*width and h<.95*height and bool(m['segmentation'][height//2,width//2]):
            eligible.append((fraction,float(m['predicted_iou']),-i,i))
    return max(eligible)[-1] if eligible else None


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--video',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--latent-frames',type=int,default=39)
    args=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('SAM2 teacher requires CUDA')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();checkpoint_sha=sha256(args.checkpoint);verify_s=time.perf_counter()-start
    if checkpoint_sha!=EXPECTED_CHECKPOINT_SHA:raise ValueError('SAM2 checkpoint differs from source lock')
    torch.cuda.reset_peak_memory_stats()
    frame_dir=out/'frames';frame_dir.mkdir()
    start=time.perf_counter();pixels=[]
    with av.open(args.video) as container:
        for frame in container.decode(video=0):pixels.append(frame.to_ndarray(format='rgb24'))
    if len(pixels)!=4*args.latent_frames-3 or len(pixels)<9:raise ValueError('full expected Dense video is required')
    for i,frame in enumerate(pixels):Image.fromarray(frame).save(frame_dir/f'{i:05d}.jpg',quality=100,subsampling=0)
    decode_materialize_s=time.perf_counter()-start
    height,width=pixels[0].shape[:2]
    # Use the exact image consumed by the JPEG-backed SAM2 predictor.
    prefix_image=np.asarray(Image.open(frame_dir/'00008.jpg').convert('RGB'))
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    start=time.perf_counter()
    model=build_sam2_video_predictor('configs/sam2/sam2_hiera_l.yaml',args.checkpoint,device='cuda',apply_postprocessing=False)
    generator=SAM2AutomaticMaskGenerator(model,points_per_side=16,points_per_batch=32,min_mask_region_area=0)
    torch.cuda.synchronize();load_s=time.perf_counter()-start
    start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):candidates=generator.generate(prefix_image)
    torch.cuda.synchronize();initialization_s=time.perf_counter()-start
    chosen=select_mask(candidates,height,width)
    if chosen is None:
        (out/'result.json').write_text(json.dumps({'status':'negative','reason':'no automatic foreground mask',
            'model_load_s':load_s,'initialization_s':initialization_s,'manual_roi_used':False},indent=2)+'\n')
        return
    masks=np.zeros((len(pixels),height,width),dtype=np.bool_)
    times={}
    start=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        state=model.init_state(video_path=str(frame_dir),offload_video_to_cpu=True,offload_state_to_cpu=True,async_loading_frames=False)
        model.add_new_mask(state,frame_idx=8,obj_id=1,mask=candidates[chosen]['segmentation'])
    torch.cuda.synchronize();times['state_init_s']=time.perf_counter()-start
    for reverse,key in [(True,'completed_prefix_backward_s'),(False,'offline_forward_s')]:
        start=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for frame_id,obj_ids,logits in model.propagate_in_video(state,start_frame_idx=8,
                    max_frame_num_to_track=8 if reverse else len(pixels),reverse=reverse):
                mask=(logits[0]>0).detach().cpu().numpy().squeeze()
                if mask.shape!=(height,width):raise ValueError('unexpected mask geometry')
                masks[int(frame_id)]=mask
        torch.cuda.synchronize();times[key]=time.perf_counter()-start
    empty=np.flatnonzero(~masks.reshape(len(masks),-1).any(1)).tolist()
    pixel_patch=F.interpolate(torch.from_numpy(masks).float().unsqueeze(1),size=(30,52),mode='area').squeeze(1)
    latent_patch=align_pixel_patch_masks_to_latent(pixel_patch,latent_frames=args.latent_frames)
    torch.save({'pixel_patch_masks':pixel_patch,'latent_anchor_masks':latent_patch,'anchor_indices':torch.arange(args.latent_frames)*4,
        'mask_scope':'offline_dense_reference_teacher','causal_online':False},out/'teacher_masks.pt')
    np.savez_compressed(out/'pixel_masks.npz',masks=masks)
    for i in [8,len(pixels)//2,len(pixels)-1]:
        overlay=pixels[i].astype(float);overlay[masks[i]]=overlay[masks[i]]*.55+np.array([0,220,80])*.45
        Image.fromarray(overlay.astype(np.uint8)).save(out/f'overlay_{i:04d}.png')
    result={'status':'pass' if not empty else 'negative','scope':'automatic_prefix_initialized_full_video_oracle_teacher',
        'video_sha256':sha256(args.video),'checkpoint_sha256':checkpoint_sha,'checkpoint_verify_s':verify_s,
        'frames':len(pixels),'latent_frames':args.latent_frames,'initialization_pixel_frame':8,
        'initialization_completed_prefix_frames':9,'manual_roi_used':False,'online_method':False,
        'source_pixel_indexing_and_aligned_latent_masks_both_saved':True,'empty_frames':empty,
        'empty_masks_filled':False,'semantic_ground_truth':False,'model_load_s':load_s,
        'automatic_initialization_s':initialization_s,'cpu_decode_materialize_s':decode_materialize_s,**times,
        'frame_materialization':'JPEG quality100 subsampling0, explicitly lossy',
        'selected_area_fraction':candidates[chosen]['area']/(height*width),'gpu':torch.cuda.get_device_name(),
        'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),
        'Dense_generation_and_VAE_cost_not_included_here':True,'oracle_video_routing_not_yet_executed':True}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
