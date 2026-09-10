#!/usr/bin/env python3
"""CPU-only duration integrity and phase-aligned own-source video evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha(path):
    with path.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()


def main():
    import av
    import torch
    import numpy as np
    from PIL import Image,ImageDraw
    from adapters.longlive_sparse.history_cache import tensor_sha256
    from scripts.build_video_review_storyboards import storyboard
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--reference',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    d=json.loads((args.case/'summary.json').read_text());ref=json.loads((args.reference/'summary.json').read_text()) if args.reference else None
    assert d['status']=='pass' and d['duration_probe']['base_latent_frames']==128
    latent=torch.load(args.case/'latents.pt',weights_only=True,map_location='cpu')
    prefix_latents=None;noise_exact=None;return_noise_exact=None
    if ref:
        assert ref['status']=='pass'
        for field in ('seed','cut_scenario','assets_manifest_sha256','fixed_native_adaln_recipe','gpu'):
            assert d[field]==ref[field],field
        if ref.get('duration_probe'):
            assert d['duration_probe']['base_noise_sha256']==ref['duration_probe']['base_noise_sha256']
            if d['latent_frames']==ref['latent_frames']:assert d['noise_sha256']==ref['noise_sha256']
            if d['duration_probe'].get('noise_alignment')==ref['duration_probe'].get('noise_alignment')=='return_event':
                assert d['duration_probe']['return_noise_sha256']==ref['duration_probe']['return_noise_sha256']
                return_noise_exact=True
        else:assert d['duration_probe']['base_noise_sha256']==ref['noise_sha256']
        noise_exact=True
        prefix_latents=min(d['segments'][-1]['start_latent'],ref['segments'][-1]['start_latent'])
        old=torch.load(args.reference/'latents.pt',weights_only=True,map_location='cpu')
        assert torch.equal(latent[:,:prefix_latents],old[:,:prefix_latents]);del old
    assert tensor_sha256(latent)==d['latent_sha256']
    del latent
    target=d['duration_probe']['return_start_latent'];length=d['latent_frames'];expected=4*length-3
    keys=(188,4*target-4,4*(target+8)-4,expected-1)
    frames=[];full_hash=hashlib.sha256();prefix_hash=hashlib.sha256()
    with av.open(str(args.case/'video.mp4')) as c:
        c.streams.video[0].codec_context.thread_count=2
        assert float(c.streams.video[0].average_rate)==24
        for i,frame in enumerate(c.decode(video=0)):
            pixels=frame.to_ndarray(format='rgb24').tobytes();full_hash.update(pixels)
            if prefix_latents is not None and i<4*prefix_latents-3:prefix_hash.update(pixels)
            frames.append(frame.reformat(width=208,height=120).to_ndarray(format='rgb24'))
            if i in keys:frame.to_image().save(args.output/f'native{i}.png')
    assert len(frames)==expected==d['pixels']['frames']
    ref_prefix=hashlib.sha256()
    if ref:
        with av.open(str(args.reference/'video.mp4')) as c:
            c.streams.video[0].codec_context.thread_count=2
            for i,frame in enumerate(c.decode(video=0)):
                if i>=4*prefix_latents-3:break
                ref_prefix.update(frame.to_ndarray(format='rgb24').tobytes())
    source_end,away_end,first_end,late_end=keys
    periods={'source':(157,source_end+1),'last_away':(away_end-127,away_end+1),
             'first_return':(away_end+1,first_end+1),'late_return':(first_end+1,late_end+1)}
    for name,(a,b) in periods.items():storyboard(frames,np.linspace(a,b-1,16).round().astype(int),args.output/(name+'.png'))
    for page in range(4):storyboard(frames,np.arange(away_end-127+32*page,away_end-95+32*page),args.output/f'last_away_all{page}.png')
    panel=Image.new('RGB',(8*224,158),'white');draw=ImageDraw.Draw(panel)
    draw.text((4,2),'SOURCE | LAST AWAY | FIRST RETURN | LATE RETURN',fill='black')
    indices=(173,source_end,away_end-67,away_end,away_end+5,first_end,first_end+33,late_end)
    for col,i in enumerate(indices):
        panel.paste(Image.fromarray(frames[i]).resize((224,123)),(col*224,30));draw.text((col*224+3,17),str(i),fill='black')
    panel.save(args.output/'comparison.jpg',quality=95)
    pipeline=d['video_pipeline']
    report=dict(status='technical_pass',case=str(args.case),source=d['runner_commit'],reference=str(args.reference) if ref else None,
        actual_prefix_latents_exact=True if ref else None,compared_prefix_latents=prefix_latents,
        decoded_RGB_prefix_exact=prefix_hash.hexdigest()==ref_prefix.hexdigest() if ref else None,
        compared_prefix_pixel_frames=4*prefix_latents-3 if ref else None,base_noise_exact_to_reference=noise_exact,
        return_noise_exact_to_reference=return_noise_exact,
        actual_GPU=d['gpu'],causal_scene_memory=d.get('causal_scene_memory'),
        decoded_frames=expected,decoded_RGB_sha256=full_hash.hexdigest(),duration_probe=d['duration_probe'],
        generation_s=d['native_DiT_s'],complete_delivery_s=pipeline['complete_s'],first_packet_s=d['pixels']['first_packet_muxed_s'],
        generation_GPU_peak_bytes=d['generation_peak_allocated_bytes'],VAE_GPU_peak_bytes=d['pipeline_VAE_GPU_peak_allocated_bytes'],
        native_active_KV_bytes=d['native_positive_and_negative_KV_bytes'],
        pipeline={k:v for k,v in pipeline.items() if k!='records'},
        payload_sha256={name:sha(args.case/name) for name in ('latents.pt','video.mp4','summary.json')},
        semantic_review_complete=False,blind_human_review=False,
        limitations=['single diagnostic, not a repeated speedup','two physical GPUs charged',
            'scripted real generated extended-away history, not natural access or archive growth',
            'noise alignment is explicitly recorded; absolute mode does not match shifted return-event noise',
            'even event-aligned cross-length comparisons still differ in generated away history and absolute positions',
            'CUPTI overlap not captured for this new run'])
    (args.output/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','decoded_frames','actual_prefix_latents_exact','compared_prefix_latents','decoded_RGB_prefix_exact','complete_delivery_s')}))


if __name__=='__main__':main()
