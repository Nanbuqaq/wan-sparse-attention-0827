#!/usr/bin/env python3
"""Decode actual uninterrupted/pulse latents for descriptive visual inspection.

The fork videos contain the unchanged 30-latent prefix and the actually generated
nine-latent suffix. Never splice baseline frames AFTER an intervention suffix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import av
from PIL import Image, ImageDraw
import torch
from torchvision.io import write_video

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.video_decode import decode_latents_chunked_exact, expected_pixel_frames


def video_frame_count(path):
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def build_fork_latents(baseline, suffix, *, start=30):
    if baseline.shape[0]!=1 or suffix.shape[0]!=1 or baseline.shape[2:]!=suffix.shape[2:]:
        raise ValueError('incompatible fork geometry')
    if start < 0 or start+suffix.shape[1] > baseline.shape[1]:
        raise ValueError('fork outside baseline length')
    return torch.cat((baseline[:,:start],suffix),dim=1)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('real CUDA required for VAE')
    root=Path(args.input)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    terminal=json.loads((root/'terminal.json').read_text())
    if terminal['status']!='pass' or not terminal['pulses'][0]['bitwise_latent_match']:
        raise ValueError('passing no-op gate required before visual comparison')
    base_source=Path(os.environ['LONGLIVE_BASE_SOURCE'])
    sys.path.insert(0,str(base_source))
    from utils.wan_wrapper import WanVAEWrapper
    vae=WanVAEWrapper().eval().requires_grad_(False).to(device='cuda',dtype=torch.bfloat16)
    baseline=torch.load(root/'uninterrupted_latents.pt',map_location='cpu',weights_only=True)
    policies=['reference','oldest','newest','random']
    indices=[116,120,128,140,152]
    canvas=Image.new('RGB',(312*len(indices),204*len(policies)),color='white')
    draw=ImageDraw.Draw(canvas)
    reference_pixels=None
    records=[]
    for row,policy in enumerate(policies):
        suffix=torch.load(root/f'pulse_{policy}_latents.pt',map_location='cpu',weights_only=True)
        latent=build_fork_latents(baseline,suffix)
        video=decode_latents_chunked_exact(vae,latent.to('cuda'),chunk_size=6)
        pixels=((video*.5+.5).clamp(0,1)*255).permute(0,1,3,4,2)[0].to(torch.uint8)
        if policy=='reference':
            reference_pixels=pixels.clone()
        if not torch.equal(reference_pixels[:117],pixels[:117]):
            raise RuntimeError('VAE fork changed the common causal pixel prefix')
        path=out/f'{policy}.mp4'
        write_video(str(path),pixels,fps=16)
        count=video_frame_count(path)
        if count!=expected_pixel_frames(latent.shape[1]):
            raise RuntimeError('encoded/decoded video frame count mismatch')
        rmse=[]
        for start in (117,129,141):
            delta=(pixels[start:start+12].float()-reference_pixels[start:start+12].float())/255.
            rmse.append(float(delta.square().mean().sqrt()))
        for column,index in enumerate(indices):
            tile=Image.fromarray(pixels[index].numpy()).resize((312,180))
            canvas.paste(tile,(column*312,row*204+24))
            draw.text((column*312+4,row*204+5),f'{policy} | pixel {index}',fill='black')
        records.append({'policy':policy,'video':str(path.resolve()),'decoded_frames':count,
                        'video_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'common_prefix_raw_pixels_exact':True,'per_chunk_pixel_rmse':rmse})
        del video,pixels
    canvas.save(out/'pulse_comparison.png')
    result={'status':'pass','scope':'descriptive_prefix_plus_actual_suffix_visualization',
            'source_probe':str(root.resolve()),'generator_source_commit':terminal['source_commit'],
            'records':records,'absolute_quality_ranking':None,
            'note':'pixel RMSE is relative divergence, not perceptual/semantic quality'}
    (out/'render_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
