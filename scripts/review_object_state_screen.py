#!/usr/bin/env python3
"""CPU artifact/prefix audit and boards; semantic source validity remains review."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback


def main():
    import av
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    cases=[];prefix={}
    selections=dict(source=[160,164,168,172,176,180,184,188],
                    away=np.linspace(253,380,16,dtype=int).tolist(),
                    returned=np.linspace(381,508,16,dtype=int).tolist(),
                    whole=np.linspace(0,508,16,dtype=int).tolist())
    wanted=set(sum(selections.values(),[]))
    for seed in (20260925,20260926):
        for obj in ('chest','envelope'):
            for mode in ('revisit','visible_control'):
                case_id=f'{obj}_{mode}';root=args.root/f'seed{seed}'/case_id
                row=dict(seed=seed,case=case_id)
                try:
                    d=json.loads((root/'summary.json').read_text())
                    if d['status']!='pass':raise ValueError('generation did not technically pass')
                    assert d['latent_shape']==[1,128,48,44,80] and d['pixel_frames']==509
                    assert d['object_state_dense_screen']['formal_holdout'] is False
                    latent=torch.load(root/'latents.pt',map_location='cpu',weights_only=True)
                    assert list(latent.shape)==d['latent_shape'] and torch.isfinite(latent).all()
                    prefix[(seed,case_id)]=latent[:,:48].clone();del latent
                    images={};digest=hashlib.sha256();count=0
                    with av.open(str(root/'video.mp4')) as video:
                        video.streams.video[0].codec_context.thread_count=2
                        for i,frame in enumerate(video.decode(video=0)):
                            rgb=frame.to_ndarray(format='rgb24')
                            if i<189:digest.update(rgb.tobytes())
                            if i in wanted:images[i]=Image.fromarray(rgb).resize((256,141))
                            count+=1
                    assert count==509 and set(images)==wanted
                    destination=args.output/f'{obj}_s{seed}_{mode}';destination.mkdir()
                    for name,indices in selections.items():
                        board=Image.new('RGB',(4*256,((len(indices)+3)//4)*165),'white');draw=ImageDraw.Draw(board)
                        for j,index in enumerate(indices):
                            x,y=(j%4)*256,(j//4)*165;draw.text((x+5,y+3),f'pixel {index}',fill='black');board.paste(images[index],(x,y+24))
                        board.save(destination/f'{name}.png')
                    row.update(status='pass',frames=count,noise_sha256=d['noise_sha256'],
                               prefix_RGB_sha256=digest.hexdigest(),summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
                               source_validity='pending_descriptive_review',boards=str(destination))
                except Exception:row.update(status='fail',traceback=traceback.format_exc())
                cases.append(row);print(json.dumps({k:v for k,v in row.items() if k!='traceback'}),flush=True)
    pairs=[]
    for seed in (20260925,20260926):
        for obj in ('chest','envelope'):
            a=next(r for r in cases if r['seed']==seed and r['case']==obj+'_revisit')
            b=next(r for r in cases if r['seed']==seed and r['case']==obj+'_visible_control')
            passed=a['status']==b['status']=='pass'
            if passed:
                passed=(a['noise_sha256']==b['noise_sha256'] and a['prefix_RGB_sha256']==b['prefix_RGB_sha256']
                        and torch.equal(prefix[(seed,a['case'])],prefix[(seed,b['case'])]))
            pairs.append(dict(seed=seed,object=obj,status='pass' if passed else 'fail',
                              actual_first48_latents_and_first189_decoded_pixels_equal=bool(passed)))
    result=dict(status='pass' if all(r['status']=='pass' for r in cases+pairs) else 'fail',
                cases=cases,pairs=pairs,technical_and_prefix_only=True,semantic_review_pending=True)
    (args.output/'technical_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    links=''.join(f'<h2>{r["case"]} / {r["seed"]}</h2><p>{r["status"]}; source validity pending</p>'
                  +(''.join(f'<h3>{part}</h3><img src="{Path(r["boards"]).name}/{part}.png">' for part in ('source','away','returned','whole')) if r['status']=='pass' else '') for r in cases)
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Independent object-state source screen</title><style>body{font:17px system-ui;max-width:1100px;margin:30px auto}img{max-width:100%}</style><h1>Dense-only source feasibility; no memory method ranking</h1>'+links)


if __name__=='__main__':main()
