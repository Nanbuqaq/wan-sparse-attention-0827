#!/usr/bin/env python3
"""Full decode audit and fixed temporal coverage boards; no automatic quality verdict."""
import argparse,hashlib,json
from pathlib import Path
import av
import numpy as np
from PIL import Image,ImageDraw
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--include',nargs='+');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);rows=[]
    for case in sorted(args.batch.iterdir()):
        if args.include is not None and case.name not in args.include:continue
        if not (case/'summary.json').exists():continue
        d=json.loads((case/'summary.json').read_text())
        if d['status']!='pass':rows.append({'case':case.name,'status':d['status']});continue
        out=args.output/case.name;out.mkdir();sample_ids=list(range(0,509,8));sample_ids[-1]=508
        boards=[Image.new('RGB',(1280,1600),'white') for _ in range(2)];digest=hashlib.sha256();count=0;previous=None;motion=[]
        with av.open(str(case/'video.mp4')) as container:
            container.streams.video[0].codec_context.thread_count=2
            for frame in container.decode(video=0):
                rgb=frame.to_ndarray(format='rgb24');digest.update(memoryview(rgb))
                if previous is not None:motion.append(float(np.abs(rgb.astype(np.int16)-previous.astype(np.int16)).mean()))
                previous=rgb
                if count in sample_ids:
                    n=sample_ids.index(count);page,pos=divmod(n,32);y,x=divmod(pos,4);board=boards[page]
                    board.paste(Image.fromarray(rgb).resize((320,176)),(x*320,y*200+24));ImageDraw.Draw(board).text((x*320+3,y*200+3),str(count),fill='black')
                if count in (0,60,124,157,188,189,253,380,381,444,508):Image.fromarray(rgb).save(out/f'frame{count:04d}.png')
                count+=1
        if count!=509:raise ValueError('incomplete full-size video')
        for i,b in enumerate(boards):b.save(out/f'timeline{i}.jpg',quality=92)
        latents=torch.load(case/'latents.pt',map_location='cpu',weights_only=True)
        h=hashlib.sha256();h.update(str(latents.dtype).encode());h.update(json.dumps(list(latents.shape)).encode());h.update(latents.contiguous().view(torch.uint8).numpy().tobytes())
        if h.hexdigest()!=d['latent_sha256']:raise ValueError('actual latent hash differs')
        row=dict(case=case.name,status='pass',frames=count,decoded_sha256=digest.hexdigest(),noise_sha256=d['noise_sha256'],
            GPU=d['gpu'],generation_s=d['native_DiT_s'],delivery_s=d['video_pipeline']['complete_s'],
            frame_RGB_change_mean=float(np.mean(motion)),frame_RGB_change_p95=float(np.quantile(motion,.95)),
            note='pixel change is not semantic motion or state validity; fixed timeline and original frames require review')
        rows.append(row);print(json.dumps(row),flush=True)
    (args.output/'payload_audit.json').write_text(json.dumps({'rows':rows,'semantic_review_pending':True},indent=2)+'\n')


if __name__=='__main__':main()
