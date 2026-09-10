#!/usr/bin/env python3
"""Past-source-only oracle geometry feasibility; never an automatic online method."""
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
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_sam2_oracle_masks import EXPECTED_CHECKPOINT_SHA,sha256
from adapters.longlive_sparse.history_cache import tensor_sha256


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--latents',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bbox',nargs=4,type=float,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    verify=time.perf_counter();checkpoint_sha=sha256(args.checkpoint)
    if checkpoint_sha!=EXPECTED_CHECKPOINT_SHA:raise ValueError('SAM2 checkpoint source lock differs')
    verify_s=time.perf_counter()-verify;began=time.perf_counter();pixels=[];digest=hashlib.sha256()
    with av.open(str(args.video)) as c:
        c.streams.video[0].codec_context.thread_count=2
        for index,frame in enumerate(c.decode(video=0)):
            if index>=189:break
            if index>=157:
                array=frame.to_ndarray(format='rgb24');pixels.append(array);digest.update(array.tobytes())
    if len(pixels)!=32 or pixels[0].shape!=(704,1280,3):raise ValueError('original native source40–47 pixels157–188 required')
    decode_s=time.perf_counter()-began
    # The reference hash covers only the past source latent slice.
    latent=torch.load(args.latents,map_location='cpu',weights_only=True,mmap=True)
    source_latent_sha=tensor_sha256(latent[:,40:48]);del latent
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
    model=build_sam2('configs/sam2/sam2_hiera_l.yaml',str(args.checkpoint),device='cuda',apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model);torch.cuda.synchronize();load_s=time.perf_counter()-began
    masks=[];records=[];began=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for index,array in enumerate(pixels):
            predictor.set_image(array)
            proposed,scores,_=predictor.predict(box=np.asarray(args.bbox,dtype=np.float32),multimask_output=True)
            chosen=int(np.argmax(scores));mask=np.asarray(proposed[chosen],dtype=np.bool_)
            masks.append(mask);records.append(dict(pixel_frame=157+index,chosen=chosen,scores=np.asarray(scores).tolist(),area_pixels=int(mask.sum())))
    torch.cuda.synchronize();mask_s=time.perf_counter()-began
    stacked=np.stack(masks);tokens=stacked.reshape(8,4,22,32,40,32).any(axis=(1,3,5))
    ids=np.flatnonzero(tokens.reshape(-1)).astype(np.int64)
    np.savez_compressed(args.output/'source_masks.npz',pixel_masks=stacked,token_masks=tokens,source_token_indices=ids)
    payload=dict(schema='native_past_source_mask_v1',source_start=40,source_end=48,source_latent_sha256=source_latent_sha,
        source_pixel_sha256=digest.hexdigest(),indices=torch.from_numpy(ids),token_grid=[22,40],
        scope='manual-box past-source geometry oracle; no automatic online or optimal-KV claim',manual_bbox=args.bbox)
    torch.save(payload,args.output/'source_mask_indices.pt')
    for index in (0,15,31):
        overlay=pixels[index].astype(np.float32);overlay[stacked[index]]=overlay[stacked[index]]*.6+np.array([0,220,100])*.4
        Image.fromarray(overlay.astype(np.uint8)).save(args.output/f'source_overlay{157+index}.png')
    result=dict(status='mask_feasibility_complete',checkpoint_sha256=checkpoint_sha,checkpoint_verify_s=verify_s,
        source_pixel_sha256=digest.hexdigest(),source_latent_sha256=source_latent_sha,manual_bbox=args.bbox,
        source_pixels_only=True,return_or_future_pixels_supplied_to_predictor=False,semantic_ground_truth=False,
        source_token_indices=len(ids),source_total_tokens=7040,source_fraction=len(ids)/7040,
        tokens_per_source_frame=tokens.sum(axis=(1,2)).tolist(),fits_quarter_raw_budget=len(ids)<=1760,
        mask_selection='highest SAM2 predicted IoU per past source frame; no quality-dependent mask choice',
        token_mapping='any foreground pixel across4 decoded frames and32x32 spatial patch; no truncation to force budget',
        CPU_decode_s=decode_s,model_load_s=load_s,all_source_mask_wall_s=mask_s,GPU=torch.cuda.get_device_name(0),
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),records=records,
        complete_online_VAE_segmentation_routing_cost_not_measured=True,video_intervention_not_yet_run=True)
    (args.output/'mask_feasibility.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
