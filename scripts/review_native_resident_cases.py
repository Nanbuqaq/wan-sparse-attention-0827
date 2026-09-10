#!/usr/bin/env python3
"""Complete decode and own-source review material for a native/candidate pair."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import av
import numpy as np
from PIL import Image,ImageDraw
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_video_review_storyboards import storyboard


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    summaries=[json.loads((path/'summary.json').read_text()) for path in (args.baseline,args.candidate)]
    assert all(d['status']=='pass' for d in summaries)
    for key in ('seed','latent_shape','noise_sha256','prompts_per_block','fixed_native_adaln_recipe','assets_manifest_sha256'):
        assert summaries[0][key]==summaries[1][key],key
    latents=[torch.load(path/'latents.pt',map_location='cpu',weights_only=True) for path in (args.baseline,args.candidate)]
    different=(latents[0]!=latents[1]).reshape(128,-1).any(-1).nonzero().flatten().tolist()
    panels=[];rows=[]
    for name,path,d in zip(('native','candidate'),(args.baseline,args.candidate),summaries):
        frames=[];keys={};digest=hashlib.sha256()
        with av.open(str(path/'video.mp4')) as container:
            container.streams.video[0].codec_context.thread_count=2
            for i,frame in enumerate(container.decode(video=0)):
                rgb=frame.to_ndarray(format='rgb24');digest.update(rgb.tobytes())
                if i in (188,380,412,508):
                    img=Image.fromarray(rgb);keys[i]=img
                    img.save(args.output/f'{name}_native_frame{i}.png')
                frames.append(frame.reformat(width=208,height=120).to_ndarray(format='rgb24'))
        assert len(frames)==509
        for label,start,end in [('source',157,189),('away',253,381),('first_return',381,413),('late',413,509)]:
            storyboard(frames,np.linspace(start,end-1,16).round().astype(int),args.output/f'{name}_{label}.png')
        for page in range(4):
            storyboard(frames,np.arange(253+32*page,285+32*page),args.output/f'{name}_away_all_{page}.png')
        panel=Image.new('RGB',(4*640,380),'white');draw=ImageDraw.Draw(panel)
        draw.text((8,8),name+' | own source188 | away380 | first return412 | late508',fill='black')
        for i,index in enumerate((188,380,412,508)):panel.paste(keys[index].resize((640,352)),(i*640,28))
        panels.append(panel)
        h=d.get('resident_history',{});calls=h.get('rows',[])
        rows.append(dict(name=name,case=str(path),frames=len(frames),decoded_RGB_sha256=digest.hexdigest(),
            latent_sha256=d['latent_sha256'],native_DiT_s=d['native_DiT_s'],native_VAE_s=d['native_VAE_s'],
            peak_GPU_allocated_bytes=d['generation_peak_allocated_bytes'],
            history_summary_GPU_bytes=h.get('summary_GPU_peak_bytes',0),
            aggregate_executed_density=(sum(c['logical_pairs'] for c in calls)/sum(c['full_native_pairs'] for c in calls)) if calls else 1.,
            dispatches=len(calls),reused_routes=sum(c['route_reused'] for c in calls)))
    combined=Image.new('RGB',(2560,760),'white')
    for i,panel in enumerate(panels):combined.paste(panel,(0,i*380))
    combined.save(args.output/'comparison.jpg',quality=95)
    result=dict(status='technical_review_material_complete',rows=rows,
        first_different_latent_frame=different[0] if different else None,
        prefix_before_first_difference_exact=True,semantic_review_complete=False,
        single_pair_not_repeated_speedup=True,own_source_required=True)
    (args.output/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
