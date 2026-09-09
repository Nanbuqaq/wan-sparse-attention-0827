#!/usr/bin/env python3
"""Prefix audit and review for cut text versus RoPE ablations."""
import argparse
import hashlib
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
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--available-only',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);groups=[];body=[]
    for lane in range(2):
        control=args.controls/f'lane{lane}/settled_bead_visible_control'
        original=json.loads((control/'summary.json').read_text());base=torch.load(control/'latents.pt',map_location='cpu',weights_only=True)
        specs=[('native_cut',control)]+[(arm,args.root/f'lane{lane}'/arm) for arm in ('strip_words','freeze_rope','strip_words_freeze_rope')]
        panels=[];rows=[];prefix=None
        for arm,root in specs:
            path=root/'summary.json'
            if not path.exists() and args.available_only:rows.append(dict(arm=arm,status='pending'));continue
            d=json.loads(path.read_text())
            if d['status']!='pass':rows.append(dict(arm=arm,status='fail',error=d.get('traceback')));continue
            assert d['seed']==20260919+lane and d['noise_sha256']==original['noise_sha256']
            assert d['latent_shape']==[1,128,48,44,80] and d['expected_scene_cut_block_indices']==[2,6,12]
            assert [e['completed_latent'] for e in d['native_shot_pin_events']]==[24,56,104]
            latent=torch.load(root/'latents.pt',map_location='cpu',weights_only=True)
            assert torch.equal(latent[:,:48],base[:,:48])
            if arm!='native_cut':
                assert d['cut_component_ablation']==arm
                if 'freeze_rope' in arm:
                    assert d['rope_phase_ablation']['phase']==8
                    assert all(e['used']==8 for e in d['rope_phase_ablation']['events'])
                if 'strip_words' in arm:assert d['condition_text_aliases']
            frames=[];h=hashlib.sha256();boards={}
            with av.open(str(root/'video.mp4')) as container:
                for i,frame in enumerate(container.decode(video=0)):
                    frames.append(frame.reformat(width=208,height=120).to_ndarray(format='rgb24'))
                    if i<189:h.update(frame.to_ndarray(format='rgb24').tobytes())
                    if i in (188,220,380,412,508):
                        target=args.output/f'lane{lane}__{arm}__frame{i}.png';frame.to_image().save(target);boards[f'native{i}']=target.name
            assert len(frames)==509
            if prefix is None:prefix=h.hexdigest()
            assert prefix==h.hexdigest()
            for name,(a,b) in {'source':(157,189),'first_cut':(189,221),'middle':(253,381),'last_cut':(381,413),'late':(445,509),
                               'quarter1':(0,125),'quarter2':(125,253),'quarter3':(253,381),'quarter4':(381,509)}.items():
                target=args.output/f'lane{lane}__{arm}__{name}.png';storyboard(frames,np.linspace(a,b-1,16).round().astype(int),target);boards[name]=target.name
            panel=Image.new('RGB',(1792,150),'white');draw=ImageDraw.Draw(panel);draw.text((4,2),f'{20260919+lane} {arm}',fill='black')
            for col,i in enumerate((156,188,197,220,313,380,412,508)):
                panel.paste(Image.fromarray(frames[i]).resize((224,123)),(224*col,25));draw.text((224*col+3,12),str(i),fill='black')
            panels.append(panel);rows.append(dict(arm=arm,status='technical_pass',actual_pre48_and_decoded_prefix_exact=True,boards=boards,
                summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),condition_aliases=d.get('condition_text_aliases'),
                rope_ablation=d.get('rope_phase_ablation'),native_pin_events=d['native_shot_pin_events']))
        canvas=Image.new('RGB',(1792,150*len(panels)),'white')
        for i,panel in enumerate(panels):canvas.paste(panel,(0,150*i))
        canvas.save(args.output/f'lane{lane}__comparison.jpg',quality=95)
        groups.append(dict(lane=lane,cases=rows));body.append(f'<h2>seed{20260919+lane}</h2><img src="lane{lane}__comparison.jpg">')
        for row in rows:body.append('<p>'+row['arm']+' ('+row['status']+'): '+' · '.join(f'<a href="{v}">{k}</a>' for k,v in row.get('boards',{}).items())+'</p>')
    (args.output/'review_evidence.json').write_text(json.dumps(dict(groups=groups,available_snapshot=args.available_only,
        semantic_review_complete=False,original_controls_reused=True),indent=2)+'\n')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Cut文字与RoPE分解</title><style>body{font:17px system-ui;max-width:1800px;margin:24px auto;padding:0 20px}img{width:100%}</style><h1>Cut文字 × RoPE相位：可见状态控制</h1><p>实际pre48 latent与前189张原分辨率decodedRGB相同；raw cut检测和native pin事件保持。</p>'+''.join(body))
    print(json.dumps(dict(groups=2,available_snapshot=args.available_only)))


if __name__=='__main__':main()
