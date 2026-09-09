#!/usr/bin/env python3
"""Own-source/state review with reused exact-prefix controls, not a quality scorer."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import av
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from scripts.build_video_review_storyboards import storyboard
    from scripts.review_native_memory_study import native_review_indices
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--available-only',action='store_true')
    p.add_argument('--pinned',type=Path);p.add_argument('--remat',type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    specs=[(name,args.controls/'lane1'/arm) for name,arm in (('none','none'),('shot','shot'),('keep_initial_reset','reset_reveal'))]
    specs += [(f'{policy}__{mode}',args.root/policy/mode) for policy in ('source_only','source_repeat') for mode in ('raw_reveal','raw_away')]
    if args.pinned:specs += [(f'source_repeat_pinned__{mode}',args.pinned/mode) for mode in ('raw_reveal','raw_away')]
    if args.remat:specs += [(f'isolated_reencode__{condition}',args.remat/condition) for condition in ('past','current')]
    base=torch.load(args.controls/'lane1/none/latents.pt',map_location='cpu',weights_only=True)
    rows=[];panels=[];prefix_reference=None
    for name,root in specs:
        summary_path=root/'summary.json'
        if not summary_path.exists() and args.available_only:
            rows.append(dict(case=name,status='pending'));continue
        d=json.loads(summary_path.read_text())
        if d['status']!='pass':rows.append(dict(case=name,status=d['status'],error=d.get('traceback')));continue
        latent=torch.load(root/'latents.pt',map_location='cpu',weights_only=True)
        if not torch.equal(base[:,:96],latent[:,:96]):raise ValueError('pre96 changed in '+name)
        frames=[];prefix=hashlib.sha256();boards={}
        with av.open(str(root/'video.mp4')) as container:
            for i,frame in enumerate(container.decode(video=0)):
                frames.append(frame.reformat(width=208,height=120).to_ndarray(format='rgb24'))
                if i<381:prefix.update(frame.to_ndarray(format='rgb24').tobytes())
                if i in native_review_indices()[0]:
                    dest=args.output/f'{name}__frame{i}.png';frame.to_image().save(dest);boards[f'native{i}']=dest.name
        if len(frames)!=509:raise ValueError('truncated original video')
        if prefix_reference is None:prefix_reference=prefix.hexdigest()
        if prefix.hexdigest()!=prefix_reference:raise ValueError('decoded pre-return pixels changed in '+name)
        for part,(start,end) in {'source':(157,189),'away':(253,381),'first_return':(381,413),'late_return':(413,509),
                                 'quarter1':(0,125),'quarter2':(125,253),'quarter3':(253,381),'quarter4':(381,509)}.items():
            dest=args.output/f'{name}__{part}.png'
            storyboard(frames,np.linspace(start,end-1,16).round().astype(int),dest);boards[part]=dest.name
        panel=Image.new('RGB',(1792,150),'white');draw=ImageDraw.Draw(panel);draw.text((4,2),name,fill='black')
        for col,i in enumerate(native_review_indices()[1]):
            panel.paste(Image.fromarray(frames[i]).resize((224,123)),(col*224,25));draw.text((col*224+3,12),str(i),fill='black')
        panels.append(panel)
        rows.append(dict(case=name,status='technical_pass',pre96_latent_and_decoded_pixels_exact=True,
            source_summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),boards=boards,
            ledger=d['episode_memory']['ledger'],installation=d['episode_memory']['installation'],
            shapes=d['episode_memory'].get('observed_return_attention_shapes'),native_DiT_s=d['native_DiT_s'],VAE_s=d['native_VAE_s']))
    if panels:
        canvas=Image.new('RGB',(1792,150*len(panels)),'white')
        for i,panel in enumerate(panels):canvas.paste(panel,(0,150*i))
        canvas.save(args.output/'comparison.jpg',quality=95)
    (args.output/'review_evidence.json').write_text(json.dumps(dict(cases=rows,available_snapshot=args.available_only,
        semantic_review_complete=False,privileged_single_seed_study=True),indent=2)+'\n')
    body=['<!doctype html><meta charset="utf-8"><title>初始状态退出与源寿命</title><style>body{font:16px system-ui;max-width:1800px;margin:24px auto;padding:0 20px}img{width:100%}</style><h1>初始状态退出与源寿命：新控制＋既有控制</h1><p>所有已有结果的pre96 latent和前381张解码RGB均精确相同。source_repeat是显式重复逻辑边，另计D2D；不是padding。单seed、实验指定源，不是自动在线方法。</p><img src="comparison.jpg">']
    for row in rows:
        body.append('<p>'+html.escape(row['case'])+' ('+row['status']+'): '+' · '.join(f'<a href="{path}">{name}</a>' for name,path in row.get('boards',{}).items())+'</p>')
    (args.output/'index.html').write_text(''.join(body));print(json.dumps(dict(rows=len(rows),available_snapshot=args.available_only)))


if __name__=='__main__':main()
