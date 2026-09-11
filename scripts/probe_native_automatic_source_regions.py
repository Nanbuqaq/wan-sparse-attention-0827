#!/usr/bin/env python3
"""Automatic proposals from one owned past source frame, with no manual box input."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import av
import numpy as np
from PIL import Image,ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_sam2_oracle_masks import EXPECTED_CHECKPOINT_SHA,sha256
from adapters.longlive_sparse.history_cache import tensor_sha256


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('real GPU required')
    began=time.perf_counter();checkpoint_sha=sha256(args.checkpoint)
    if checkpoint_sha!=EXPECTED_CHECKPOINT_SHA:raise ValueError('SAM2 checkpoint differs from locked source')
    verify_s=time.perf_counter()-began
    summary=json.loads((args.case/'summary.json').read_text())
    if summary['status']!='pass' or summary['latent_shape']!=[1,128,48,44,80]:
        raise ValueError('qualified original-resolution owned source required')
    began=time.perf_counter();pixels=None
    with av.open(str(args.case/'video.mp4')) as container:
        container.streams.video[0].codec_context.thread_count=2
        for index,frame in enumerate(container.decode(video=0)):
            if index==157:
                pixels=frame.to_ndarray(format='rgb24');break
    if pixels is None or pixels.shape!=(704,1280,3):raise ValueError('past source frame157 missing')
    decode_s=time.perf_counter()-began
    latent=torch.load(args.case/'latents.pt',weights_only=True,map_location='cpu',mmap=True)
    source_sha=tensor_sha256(latent[:,40:48]);del latent
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    torch.cuda.reset_peak_memory_stats();began=time.perf_counter()
    model=build_sam2('configs/sam2/sam2_hiera_l.yaml',str(args.checkpoint),device='cuda',apply_postprocessing=False)
    generator=SAM2AutomaticMaskGenerator(model,points_per_side=16,points_per_batch=32,
        pred_iou_thresh=.8,stability_score_thresh=.95,crop_n_layers=0,min_mask_region_area=0,
        output_mode='binary_mask')
    torch.cuda.synchronize();load_s=time.perf_counter()-began;began=time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):proposals=generator.generate(pixels)
    torch.cuda.synchronize();proposal_s=time.perf_counter()-began
    peak=torch.cuda.max_memory_allocated();rows=[];masks=[];token_masks=[]
    for index,proposal in enumerate(proposals):
        mask=np.asarray(proposal['segmentation'],dtype=np.bool_)
        token=mask.reshape(22,32,40,32).any(axis=(1,3))
        masks.append(mask);token_masks.append(token)
        rows.append(dict(id=index,pixel_area=int(mask.sum()),spatial_tokens=int(token.sum()),
            bbox=proposal['bbox'],predicted_iou=proposal['predicted_iou'],
            stability_score=proposal['stability_score'],point_coords=proposal['point_coords'],
            crop_box=proposal['crop_box']))
    arrays=np.stack(masks) if masks else np.empty((0,704,1280),dtype=np.bool_)
    tokens=np.stack(token_masks) if token_masks else np.empty((0,22,40),dtype=np.bool_)
    np.savez_compressed(args.output/'all_source_proposals.npz',pixel_masks=arrays,token_masks=tokens)
    Image.fromarray(pixels).save(args.output/'source157.png')
    for page in range((len(rows)+11)//12):
        board=Image.new('RGB',(4*320,3*205),'white');draw=ImageDraw.Draw(board)
        for slot,index in enumerate(range(page*12,min(len(rows),(page+1)*12))):
            overlay=pixels.astype(np.float32);overlay[arrays[index]]=overlay[arrays[index]]*.5+np.array([0,230,90])*.5
            x,y=(slot%4)*320,(slot//4)*205
            board.paste(Image.fromarray(overlay.astype(np.uint8)).resize((320,176)),(x,y+29))
            draw.text((x+3,y+3),f"id{index} tokens={rows[index]['spatial_tokens']} iou={rows[index]['predicted_iou']:.3f}",fill='black')
        board.save(args.output/f'proposals_page{page}.jpg',quality=95)
    report=dict(status='automatic_proposals_complete',source_case=str(args.case),source_pixel_frame=157,
        source_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),source_latent_sha256=source_sha,
        checkpoint_sha256=checkpoint_sha,checkpoint_verify_CPU_s=verify_s,CPU_decode_s=decode_s,
        model_load_s=load_s,automatic_proposals_wall_s=proposal_s,peak_GPU_allocated_bytes=peak,
        GPU=torch.cuda.get_device_name(),proposals=rows,proposal_count=len(rows),
        mask_CPU_payload_bytes=arrays.nbytes+tokens.nbytes,
        parameters=dict(points_per_side=16,points_per_batch=32,pred_iou_thresh=.8,
            stability_score_thresh=.95,crop_n_layers=0,min_mask_region_area=0),
        manual_box_or_mask_input=False,return_or_future_pixel_input=False,semantic_region_selection=False,
        limitations=['one source pixel frame only; no temporal tracking or8-frame token ownership established',
            'automatic proposal capacity is not an online memory selector or a quality result',
            'external VAE/segmentation scheduling and complete online delivery cost not measured',
            'overlapping proposals are not a disjoint source partition'])
    (args.output/'automatic_proposals.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='proposals'}),flush=True)


if __name__=='__main__':main()
