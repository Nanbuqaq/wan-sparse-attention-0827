#!/usr/bin/env python3
"""Dense-only new-state screen, with a matched always-visible control."""
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
    p.add_argument('--positive-stop',action='store_true');p.add_argument('--reference-root',type=Path);args=p.parse_args()
    if args.positive_stop and args.reference_root is None:p.error('positive-stop requires the unchanged negative-wording controls')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);groups=[]
    for lane,seed in enumerate((20260923,20260924)):
        rows=[];reference=None;noise=None;prefix=None;panels=[]
        scenarios=('blue_canvas_positive_stop_revisit','blue_canvas_positive_stop_visible_control') if args.positive_stop else ('blue_canvas_revisit','blue_canvas_visible_control')
        for scenario in scenarios:
            root=args.root/f'lane{lane}'/scenario;path=root/'summary.json';d=json.loads(path.read_text())
            if d['status']!='pass':rows.append(dict(scenario=scenario,status='fail',summary=str(path)));continue
            assert d['seed']==seed and d['latent_shape']==[1,128,48,44,80]
            assert d['episode_memory_mode'] is None and not d.get('causal_scene_memory_enabled',False)
            assert d['expected_scene_cut_block_indices']==[2,8,12]
            assert [p['completed_latent'] for p in d['native_shot_pin_events']]==[24,72,104]
            latent=torch.load(root/'latents.pt',map_location='cpu',weights_only=True)
            old_root=None
            if args.positive_stop:
                old_root=args.reference_root/f'lane{lane}'/scenario.replace('positive_stop_','')
                old=json.loads((old_root/'summary.json').read_text())
                assert d['noise_sha256']==old['noise_sha256']
                assert torch.equal(latent[:,:32],torch.load(old_root/'latents.pt',map_location='cpu',weights_only=True)[:,:32])
            if reference is None:reference=latent[:,:64].clone();noise=d['noise_sha256']
            assert torch.equal(reference,latent[:,:64]) and noise==d['noise_sha256']
            frames=[];digest=hashlib.sha256();prefix32_digest=hashlib.sha256();boards={}
            with av.open(str(root/'video.mp4')) as container:
                for i,frame in enumerate(container.decode(video=0)):
                    frames.append(frame.reformat(width=208,height=120).to_ndarray(format='rgb24'))
                    if i<253:digest.update(frame.to_ndarray(format='rgb24').tobytes())
                    if i<125:prefix32_digest.update(frame.to_ndarray(format='rgb24').tobytes())
                    if i in (60,124,220,252,380,412,508):
                        target=args.output/f'seed{seed}__{scenario}__native{i}.png';frame.to_image().save(target);boards[f'native{i}']=target.name
            assert len(frames)==509
            if old_root:
                old_digest=hashlib.sha256();old_count=0
                with av.open(str(old_root/'video.mp4')) as container:
                    for i,frame in enumerate(container.decode(video=0)):
                        if i>=125:break
                        old_digest.update(frame.to_ndarray(format='rgb24').tobytes());old_count+=1
                assert old_count==125 and old_digest.hexdigest()==prefix32_digest.hexdigest()
            if prefix is None:prefix=digest.hexdigest()
            assert prefix==digest.hexdigest()
            for label,(a,b) in dict(paint=(61,125),hold=(125,253),selected_source=(221,253),absence_or_visible=(285,381),
                first_return=(381,413),late=(413,509),quarter1=(0,125),quarter2=(125,253),quarter3=(253,381),quarter4=(381,509)).items():
                target=args.output/f'seed{seed}__{scenario}__{label}.png';storyboard(frames,np.linspace(a,b-1,16).round().astype(int),target);boards[label]=target.name
            for page in range(3):
                target=args.output/f'seed{seed}__{scenario}__absence_all_{page}.png'
                storyboard(frames,np.arange(285+page*32,317+page*32),target);boards[f'absence_all_{page}']=target.name
            target=args.output/f'seed{seed}__{scenario}__source_all.png';storyboard(frames,np.arange(221,253),target);boards['source_all']=target.name
            panel=Image.new('RGB',(1792,150),'white');draw=ImageDraw.Draw(panel);draw.text((4,2),f'{seed} {scenario}',fill='black')
            for col,i in enumerate((60,124,220,252,349,380,412,508)):
                panel.paste(Image.fromarray(frames[i]).resize((224,123)),(col*224,25));draw.text((col*224+3,12),str(i),fill='black')
            panels.append(panel);rows.append(dict(scenario=scenario,status='technical_pass',boards=boards,
                actual_pre64_latent_and_decoded_prefix_exact=True,decoded_prefix_sha256=prefix,
                actual_pre32_matches_original_wording=args.positive_stop,
                summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_summary=str(path),
                semantic_feasibility_not_automatically_passed=True))
        canvas=Image.new('RGB',(1792,150*max(1,len(panels))),'white')
        for i,panel in enumerate(panels):canvas.paste(panel,(0,i*150))
        canvas.save(args.output/f'seed{seed}__comparison.jpg',quality=95);groups.append(dict(seed=seed,cases=rows,paired_prefix_verified=sum(r['status']=='technical_pass' for r in rows)==2))
    (args.output/'technical_audit.json').write_text(json.dumps(dict(groups=groups,Dense_only=True,source_feasibility_review_complete=False,
        positive_stop_wording_control=args.positive_stop,memory_methods_run_on_this_workload=False),indent=2)+'\n')
    body=['<!doctype html><meta charset="utf-8"><title>蓝画布Dense筛选</title><style>body{font:17px system-ui;max-width:1800px;margin:25px auto;padding:0 20px}img{width:100%}</style><h1>新的状态类别：先Dense可行性，再验证记忆</h1><p>source56–63，离开前有更长停止保持。每seed两分支实际pre64 latent和前253张decoded RGB精确匹配；visible故意不离开。不能以Dense返回失败作为入选条件。</p>']
    for group in groups:
        body.append(f'<h2>seed {group["seed"]}</h2><img src="seed{group["seed"]}__comparison.jpg">')
        for row in group['cases']:body.append('<p>'+row['scenario']+' ('+row['status']+'): '+' · '.join(f'<a href="{v}">{k}</a>' for k,v in row.get('boards',{}).items())+'</p>')
    (args.output/'index.html').write_text(''.join(body));print(json.dumps(dict(groups=2,Dense_only=True,semantic_review_pending=True)))


if __name__=='__main__':main()
