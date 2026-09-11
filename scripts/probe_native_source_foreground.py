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


def source_token_masks(pixel_masks,source_start):
    a=max(0,4*source_start-3);b=4*(source_start+8)-3
    if len(pixel_masks)!=b-a:raise ValueError('complete source pixel window required')
    height,width=pixel_masks.shape[1:]
    if height%32 or width%32:raise ValueError('native32-pixel token alignment required')
    tokens=np.zeros((8,height//32,width//32),dtype=np.bool_)
    for index,mask in enumerate(pixel_masks):
        frame=(a+index+3)//4-source_start
        tokens[frame]|=mask.reshape(height//32,32,width//32,32).any(axis=(1,3))
    return tokens


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--latents',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bbox',nargs=4,type=float,required=True)
    p.add_argument('--raw-source-windows',type=Path);p.add_argument('--archive-version',type=int)
    p.add_argument('--box-provenance',type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    verify=time.perf_counter();checkpoint_sha=sha256(args.checkpoint)
    if checkpoint_sha!=EXPECTED_CHECKPOINT_SHA:raise ValueError('SAM2 checkpoint source lock differs')
    if (args.raw_source_windows is None)!=(args.archive_version is None):raise ValueError('raw source path and version must be paired')
    verify_s=time.perf_counter()-verify;began=time.perf_counter();pixels=[];digest=hashlib.sha256()
    source_start,source_end,pixel_start=40,48,157
    pixel_kind='raw_stream_before_codec' if args.raw_source_windows else 'decoded_video_rgb'
    if args.raw_source_windows:
        from adapters.longlive_sparse.native_raw_source_input import load_raw_source_window
        raw=load_raw_source_window(args.raw_source_windows,args.archive_version,json.loads((args.video.parent/'summary.json').read_text()))
        source_start,source_end,pixel_start=raw['source_start'],raw['source_end'],raw['pixel_start']
        pixels=[frame.numpy().copy() for frame in raw['pixels']]
        for array in pixels:digest.update(memoryview(array))
    else:
        with av.open(str(args.video)) as c:
            c.streams.video[0].codec_context.thread_count=2
            for index,frame in enumerate(c.decode(video=0)):
                if index>=189:break
                if index>=157:
                    array=frame.to_ndarray(format='rgb24');pixels.append(array);digest.update(array.tobytes())
    if len(pixels)!=4*source_end-3-pixel_start or pixels[0].shape!=(704,1280,3):raise ValueError('complete original-resolution source window required')
    decode_s=time.perf_counter()-began
    # The reference hash covers only the past source latent slice.
    latent=torch.load(args.latents,map_location='cpu',weights_only=True,mmap=True)
    source_latent_sha=tensor_sha256(latent[:,source_start:source_end]);del latent
    if args.raw_source_windows and source_latent_sha!=raw['source_latent_sha256']:raise ValueError('raw source and actual latent differ')
    box_provenance=None
    if args.box_provenance:
        from scripts.derive_native_source_component_box import component_box
        box_provenance=json.loads(args.box_provenance.read_text())
        proposal_path=Path(box_provenance['proposal_report'])
        proposals=json.loads(proposal_path.read_text())
        expected_box,_,_=component_box(proposals['proposals'],policy=box_provenance.get('component_policy','largest'))
        if (not box_provenance['automatic_box'] or box_provenance['manual_mask_or_return_input']
            or proposals['manual_box_or_mask_input'] or proposals['return_or_future_pixel_input']
            or sha256(proposal_path)!=box_provenance['proposal_report_sha256']
            or expected_box!=args.bbox or box_provenance['bbox']!=args.bbox
            or box_provenance.get('pixel_input_kind','decoded_video_rgb')!=pixel_kind
            or box_provenance['source_pixel_frame']!=pixel_start
            or source_latent_sha!=box_provenance['source_latent_sha256']
            or hashlib.sha256(pixels[0].tobytes()).hexdigest()!=box_provenance['source_pixel_sha256']):
            raise ValueError('automatic box source/proposal/rule provenance mismatch')
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
            chosen=int(np.argmax(scores));mask=np.asarray(proposed[chosen],dtype=np.bool_).copy()
            masks.append(mask);records.append(dict(pixel_frame=pixel_start+index,chosen=chosen,scores=np.asarray(scores).tolist(),area_pixels=int(mask.sum())))
    torch.cuda.synchronize();mask_s=time.perf_counter()-began
    stacked=np.stack(masks);tokens=source_token_masks(stacked,source_start)
    ids=np.flatnonzero(tokens.reshape(-1)).astype(np.int64)
    np.savez_compressed(args.output/'source_masks.npz',pixel_masks=stacked,token_masks=tokens,source_token_indices=ids)
    payload=dict(schema='native_past_source_mask_v1',source_start=source_start,source_end=source_end,source_latent_sha256=source_latent_sha,
        pixel_input_kind=pixel_kind,
        source_pixel_sha256=digest.hexdigest(),indices=torch.from_numpy(ids),token_grid=[22,40],
        scope=('automatic part-component box from past source; precomputed geometry diagnostic, not deployed online'
            if box_provenance else 'manual-box past-source geometry oracle; no automatic online or optimal-KV claim'),
        manual_bbox=None if box_provenance else args.bbox,prompt_bbox=args.bbox,
        box_provenance_sha256=sha256(args.box_provenance) if args.box_provenance else None)
    torch.save(payload,args.output/'source_mask_indices.pt')
    for index in (0,(len(pixels)-1)//2,len(pixels)-1):
        overlay=pixels[index].astype(np.float32);overlay[stacked[index]]=overlay[stacked[index]]*.6+np.array([0,220,100])*.4
        Image.fromarray(overlay.astype(np.uint8)).save(args.output/f'source_overlay{pixel_start+index}.png')
    result=dict(status='mask_feasibility_complete',checkpoint_sha256=checkpoint_sha,checkpoint_verify_s=verify_s,
        source_start=source_start,source_end=source_end,pixel_start=pixel_start,pixel_input_kind=pixel_kind,
        source_pixel_sha256=digest.hexdigest(),source_latent_sha256=source_latent_sha,
        manual_bbox=None if box_provenance else args.bbox,prompt_bbox=args.bbox,
        automatic_box_provenance_verified=box_provenance is not None,
        box_provenance_sha256=sha256(args.box_provenance) if args.box_provenance else None,
        source_pixels_only=True,return_or_future_pixels_supplied_to_predictor=False,semantic_ground_truth=False,
        source_token_indices=len(ids),source_total_tokens=7040,source_fraction=len(ids)/7040,
        tokens_per_source_frame=tokens.sum(axis=(1,2)).tolist(),all_source_masks_nonempty=bool(stacked.reshape(len(pixels),-1).any(1).all()),
        fits_quarter_raw_budget=0<len(ids)<=1760 and bool(stacked.reshape(len(pixels),-1).any(1).all()),
        mask_selection='highest SAM2 predicted IoU per past source frame; no quality-dependent mask choice',
        token_mapping='any foreground pixel across4 decoded frames and32x32 spatial patch; no truncation to force budget',
        CPU_input_prepare_s=decode_s,CPU_decode_s=0. if args.raw_source_windows else decode_s,
        model_load_s=load_s,all_source_mask_wall_s=mask_s,GPU=torch.cuda.get_device_name(0),
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),records=records,
        complete_online_VAE_segmentation_routing_cost_not_measured=True,video_intervention_not_yet_run=True)
    (args.output/'mask_feasibility.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
