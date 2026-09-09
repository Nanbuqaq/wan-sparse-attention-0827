#!/usr/bin/env python3
"""One full-size past-text control: actual prefix audit and source/return boards."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import av
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from scripts.build_video_review_storyboards import storyboard
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--seed',type=int,choices=(20260925,20260926),required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);result=dict(status='running',semantic_review_complete=False)
    try:
        path=args.case/'summary.json';reference_path=args.reference/'summary.json'
        d=json.loads(path.read_text());ref=json.loads(reference_path.read_text())
        assert d['status']==ref['status']=='pass' and d['seed']==ref['seed']==args.seed
        assert d['latent_shape']==ref['latent_shape']==[1,128,48,44,80]
        for key in ('noise_sha256','pre_return_latent_sha256','source_files_sha256','assets_manifest_sha256',
                    'fixed_native_adaln_recipe','triton_version','native_shot_pin_events','native_positive_and_negative_KV_bytes'):
            assert d[key]==ref[key],key
        assert d.get('causal_scene_memory') is None and d.get('episode_memory') is None
        registration=ROOT/'configs/system/native_chest_text_control.json'
        assert d['object_state_protocol']['text_control_registration_sha256']==hashlib.sha256(registration.read_bytes()).hexdigest()
        assert d['object_state_text_control']['id']=='past_settled_restatement'
        assert 'object_state_dense_screen' not in d
        assert d['prompts_per_block'][:12]==ref['prompts_per_block'][:12]
        settled=next(s['prompt'] for s in ref['segments'] if s['role']=='settled_source')
        assert all(d['prompts_per_block'][i]==ref['prompts_per_block'][i]+' '+settled for i in range(12,16))
        actual=torch.load(args.case/'latents.pt',map_location='cpu',weights_only=True)
        baseline=torch.load(args.reference/'latents.pt',map_location='cpu',weights_only=True)
        assert torch.isfinite(actual).all() and list(actual.shape)==list(baseline.shape)==d['latent_shape']
        assert torch.equal(actual[:,:96],baseline[:,:96])
        hashes=[];panels=[];boards={}
        for label,root in [('dense',args.reference),('past_text',args.case)]:
            frames=[];digest=hashlib.sha256()
            with av.open(str(root/'video.mp4')) as video:
                video.streams.video[0].codec_context.thread_count=2
                for i,frame in enumerate(video.decode(video=0)):
                    if i<381:digest.update(frame.to_ndarray(format='rgb24').tobytes())
                    frames.append(frame.reformat(width=224,height=123).to_ndarray(format='rgb24'))
                    if i in (188,385,412,460,508):frame.to_image().save(args.output/f'{label}__native{i}.png')
            assert len(frames)==509;hashes.append(digest.hexdigest())
            for name,(a,b) in dict(source=(157,189),away=(253,381),first_return=(381,413),late_return=(413,509),
                    quarter1=(0,125),quarter2=(125,253),quarter3=(253,381),quarter4=(381,509)).items():
                target=args.output/f'{label}__{name}.png';storyboard(frames,np.linspace(a,b-1,16).round().astype(int),target)
                boards[label+'__'+name]=target.name
            panel=Image.new('RGB',(1792,150),'white');draw=ImageDraw.Draw(panel);draw.text((4,2),f'{args.seed} {label}',fill='black')
            for col,i in enumerate((156,188,313,380,385,412,460,508)):
                panel.paste(Image.fromarray(frames[i]),(224*col,25));draw.text((224*col+3,12),str(i),fill='black')
            panels.append(panel)
        assert hashes[0]==hashes[1]
        comparison=Image.new('RGB',(1792,300),'white')
        for i,panel in enumerate(panels):comparison.paste(panel,(0,150*i))
        comparison.save(args.output/'comparison.jpg',quality=95)
        result.update(status='pass',seed=args.seed,actual_pre96_and_pre381_decoded_RGB_exact=True,
            extra_raw_history_KV_bank=False,native_active_KV_capacity_unchanged=True,
            different_current_condition_not_a_memory_quality_win=True,boards=boards,
            summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            reference_summary_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest())
        (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Chest past-text control</title>'
            '<style>body{font:17px system-ui;max-width:1800px;margin:25px auto}img{width:100%}</style>'
            '<h1>Dense versus explicit past-text state control</h1><p>Different current condition; not autonomous KV memory.</p>'
            '<img src="comparison.jpg">'+''.join(f'<p><a href="{path}">{name}</a></p>' for name,path in boards.items()))
    except Exception:result.update(status='fail',traceback=traceback.format_exc());raise
    finally:
        (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
