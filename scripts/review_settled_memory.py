#!/usr/bin/env python3
"""Review two predeclared settled-source seeds; preserve reused Dense controls."""
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
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--available-only',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);groups=[]
    for lane,seed in enumerate((20260919,20260920)):
        control=args.control_root/f'lane{lane}'/'settled_bead_revisit';base=json.loads((control/'summary.json').read_text())
        reference=torch.load(control/'latents.pt',map_location='cpu',weights_only=True);prefix=None;rows=[];panels=[]
        for policy,root in [('dense',control)]+[(p,args.root/f'seed{seed}'/p) for p in ('original','recent_virtual')]:
            path=root/'summary.json'
            if not path.exists() and args.available_only:rows.append(dict(policy=policy,status='pending'));continue
            d=json.loads(path.read_text())
            if d['status']!='pass':rows.append(dict(policy=policy,status='fail',summary=str(path)));continue
            assert d['seed']==seed and d['latent_shape']==[1,128,48,44,80]
            for key in ('noise_sha256','pre_return_latent_sha256','prompts_per_block','source_files_sha256','fixed_native_adaln_recipe','triton_version','native_shot_pin_events'):
                assert d[key]==base[key],key
            assert torch.equal(torch.load(root/'latents.pt',map_location='cpu',weights_only=True)[:,:96],reference[:,:96])
            if policy!='dense':
                m=d['episode_memory'];plan=m['installation']['admission_plan']
                assert d['reviewed_memory_protocol']['spec']['id']=='settled_state_v1' and d['episode_position_policy']==policy
                assert plan['source_frames']==list(range(40,48)) and plan['target_start']==96
                assert m['ledger']['demand_H2D_payload_bytes']==2595225600 and m['ledger']['demand_D2D_KV_bytes']==0
                if policy=='recent_virtual':assert plan['temporal_delta']==64
            frames=[];h=hashlib.sha256();boards={}
            for_video=root/'video.mp4'
            with av.open(str(for_video)) as container:
                for i,frame in enumerate(container.decode(video=0)):
                    frames.append(frame.reformat(width=208,height=120).to_ndarray(format='rgb24'))
                    if i<381:h.update(frame.to_ndarray(format='rgb24').tobytes())
                    if i in (156,188,380,412,508):
                        target=args.output/f'seed{seed}__{policy}__native{i}.png';frame.to_image().save(target);boards[f'native{i}']=target.name
            assert len(frames)==509
            if prefix is None:prefix=h.hexdigest()
            assert h.hexdigest()==prefix
            for label,(a,b) in dict(source_hold=(125,189),selected_source=(157,189),away=(253,381),
                first_return=(381,413),late_return=(413,509),quarter1=(0,125),quarter2=(125,253),quarter3=(253,381),quarter4=(381,509)).items():
                target=args.output/f'seed{seed}__{policy}__{label}.png'
                storyboard(frames,np.linspace(a,b-1,16).round().astype(int),target);boards[label]=target.name
            panel=Image.new('RGB',(1792,150),'white');draw=ImageDraw.Draw(panel);draw.text((4,2),f'{seed} {policy}',fill='black')
            for col,i in enumerate((156,188,313,380,385,412,445,508)):
                panel.paste(Image.fromarray(frames[i]).resize((224,123)),(col*224,25));draw.text((col*224+3,12),str(i),fill='black')
            panels.append(panel);rows.append(dict(policy=policy,status='technical_pass',boards=boards,
                source_summary=str(path),summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                actual_pre96_and_decoded_prefix_exact=True,decoded_prefix_sha256=prefix,
                ledger=d.get('episode_memory',{}).get('ledger'),native_DiT_s=d['native_DiT_s']))
        canvas=Image.new('RGB',(1792,150*len(panels)),'white')
        for i,panel in enumerate(panels):canvas.paste(panel,(0,i*150))
        canvas.save(args.output/f'seed{seed}__comparison.jpg',quality=95);groups.append(dict(seed=seed,cases=rows))
    report=dict(groups=groups,available_snapshot=args.available_only,semantic_review_complete=False,
        new_expected_video_executions=4,reused_Dense_controls=2,quality_success_not_implied_by_technical_pass=True)
    (args.output/'technical_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    body=['<!doctype html><meta charset="utf-8"><title>较静止源的历史位置</title><style>body{font:17px system-ui;max-width:1800px;margin:25px auto;padding:0 20px}img{width:100%}</style><h1>较静止源：复用Dense＋原位置＋recent位置</h1><p>每seed的实际pre96 latent和前381张decoded RGB精确匹配。source40–47来自停止命令后片段，但不是严格无运动真值。类别读取和晚段额外动作须分别判读。</p>']
    for group in groups:
        body.append(f'<h2>seed {group["seed"]}</h2><img src="seed{group["seed"]}__comparison.jpg">')
        for r in group['cases']:body.append('<p>'+r['policy']+' ('+r['status']+'): '+' · '.join(f'<a href="{v}">{k}</a>' for k,v in r.get('boards',{}).items())+'</p>')
    (args.output/'index.html').write_text(''.join(body));print(json.dumps(dict(groups=2,available_snapshot=args.available_only)))


if __name__=='__main__':main()
