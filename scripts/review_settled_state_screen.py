#!/usr/bin/env python3
"""Dense-only settled-source/visible-control screen; no automatic quality score."""
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
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--continuation',action='store_true');p.add_argument('--reference-root',type=Path);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);groups=[]
    for lane in range(2):
        rows=[];prefix=None;pixel_prefix=None;panels=[]
        scenarios=('settled_bead_nocut_anaphora','settled_bead_nocut_explicit') if args.continuation else ('settled_bead_revisit','settled_bead_visible_control')
        for scenario in scenarios:
            root=args.root/f'lane{lane}'/scenario;d=json.loads((root/'summary.json').read_text())
            if d['status']!='pass':raise ValueError('failed Dense case retained; no semantic pass')
            assert d['seed']==20260919+lane and d['latent_shape']==[1,128,48,44,80]
            expected_cuts=[2] if args.continuation else [2,6,12]
            assert d['expected_scene_cut_block_indices']==expected_cuts
            assert [e['completed_latent'] for e in d['native_shot_pin_events']]==[(i+1)*8 for i in expected_cuts]
            assert d['episode_memory_mode'] is None and d['native_KV_allocation_policy']=='CFG1_positive_only'
            latent=torch.load(root/'latents.pt',map_location='cpu',weights_only=True)
            if prefix is None:prefix=latent[:,:48].clone()
            assert torch.equal(prefix,latent[:,:48])
            if args.reference_root:
                ref_root=args.reference_root/f'lane{lane}/settled_bead_visible_control'
                ref=json.loads((ref_root/'summary.json').read_text())
                assert d['noise_sha256']==ref['noise_sha256']
                assert torch.equal(latent[:,:48],torch.load(ref_root/'latents.pt',map_location='cpu',weights_only=True)[:,:48])
            frames=[];h=hashlib.sha256();boards={}
            with av.open(str(root/'video.mp4')) as container:
                for i,frame in enumerate(container.decode(video=0)):
                    frames.append(frame.reformat(width=208,height=120).to_ndarray(format='rgb24'))
                    if i<189:h.update(frame.to_ndarray(format='rgb24').tobytes())
                    if i in (124,156,188,380,412,508):
                        path=args.output/f'lane{lane}__{scenario}__native{i}.png';frame.to_image().save(path);boards[f'native{i}']=path.name
            assert len(frames)==509
            if pixel_prefix is None:pixel_prefix=h.hexdigest()
            assert pixel_prefix==h.hexdigest()
            periods={'source_hold':(125,189),'late_source':(157,189),'middle_away_or_visible':(253,381),
                     'first_return_or_control_cut':(381,413),'late':(445,509),
                     'quarter1':(0,125),'quarter2':(125,253),'quarter3':(253,381),'quarter4':(381,509)}
            for name,(a,b) in periods.items():
                path=args.output/f'lane{lane}__{scenario}__{name}.png'
                storyboard(frames,np.linspace(a,b-1,16).round().astype(int),path);boards[name]=path.name
            for page in range(2):
                path=args.output/f'lane{lane}__{scenario}__hold_all_{page}.png'
                storyboard(frames,np.arange(125+page*32,157+page*32),path);boards[f'hold_all_{page}']=path.name
            for page in range(4):
                path=args.output/f'lane{lane}__{scenario}__middle_all_{page}.png'
                storyboard(frames,np.arange(253+page*32,285+page*32),path);boards[f'middle_all_{page}']=path.name
            panel=Image.new('RGB',(1792,150),'white');draw=ImageDraw.Draw(panel);draw.text((4,2),f's{d["seed"]} {scenario}',fill='black')
            for col,i in enumerate((92,124,156,188,313,380,412,508)):
                panel.paste(Image.fromarray(frames[i]).resize((224,123)),(col*224,25));draw.text((col*224+3,12),str(i),fill='black')
            panels.append(panel);rows.append(dict(scenario=scenario,status='technical_pass',boards=boards,
                summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
                requires_target_absence=scenario=='settled_bead_revisit',quality='pending_source_stillness_and_state_review'))
        canvas=Image.new('RGB',(1792,300),'white')
        for i,panel in enumerate(panels):canvas.paste(panel,(0,i*150))
        canvas.save(args.output/f'lane{lane}__comparison.jpg',quality=95)
        groups.append(dict(lane=lane,source_pre48_latent_and_decoded_prefix_exact=True,cases=rows))
    (args.output/'technical_audit.json').write_text(json.dumps(dict(status='pass',groups=groups,expected_cases=4,missing=0,
        Dense_only=True,new_memory_methods_not_run=True,semantic_review_complete=False),indent=2)+'\n')
    body=['<!doctype html><meta charset="utf-8"><title>静止状态与可见控制</title><style>body{font:17px system-ui;max-width:1800px;margin:24px auto;padding:0 20px}img{width:100%}</style><h1>先让状态静止，再测试离开：Dense-only筛选</h1><p>每seed的两条控制在pre48 latent和前189张解码RGB精确一致；visible control故意不离开目标。先审source40–47是否真正停止动作，再决定能否作新记忆工作负载。</p>']
    for g in groups:
        body.append(f'<h2>seed {20260919+g["lane"]}</h2><img src="lane{g["lane"]}__comparison.jpg">')
        for r in g['cases']:body.append('<p>'+r['scenario']+': '+' · '.join(f'<a href="{path}">{name}</a>' for name,path in r['boards'].items())+'</p>')
    (args.output/'index.html').write_text(''.join(body));print(json.dumps(dict(status='pass',cases=4,semantic_review_complete=False)))


if __name__=='__main__':main()
