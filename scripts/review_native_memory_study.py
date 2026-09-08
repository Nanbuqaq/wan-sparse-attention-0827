#!/usr/bin/env python3
"""Build full-video review evidence, never infer a semantic pass from fidelity."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def native_review_indices():
    # End-exclusive pixel boundary for latent f>0 is 4*f-3.
    # Frame189 starts the away shot; it is NOT a source reference.
    ends=tuple(4*f-4 for f in (48,96,104,128))
    comparison=(173,ends[0],313,ends[1],385,ends[2],445,ends[3])
    return ends,comparison


def main():
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    import av
    from scripts.build_video_review_storyboards import storyboard
    from scripts.review_revisit_videos import pixel_start
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--context-study',action='store_true')
    p.add_argument('--baseline-only',action='store_true');p.add_argument('--available-only',action='store_true');args=p.parse_args()
    if args.context_study:
        from scripts.run_native_context_study import ARMS
        from scripts.collect_native_context_study import collect
    else:
        from scripts.run_native_memory_study import ARMS
        from scripts.collect_native_memory_study import collect
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    if args.baseline_only:
        ARMS=('none',)
        terminal=dict(status='partial_baseline_feasibility_review',remaining_arms_not_audited=True)
    elif args.available_only:
        terminal=dict(status='in_progress_snapshot_not_terminal_audit',remaining_arms_not_audited=True)
    else:terminal=collect(args.root)
    groups=[];sections=[]
    for lane in range(2 if args.context_study else 4):
        panels=[];rows=[];base=None;global_latent=None
        for arm in ARMS:
            case=args.root/f'lane{lane}'/arm
            summary=case/'summary.json'
            if args.available_only and not summary.exists():
                rows.append(dict(arm=arm,status='pending_not_audited'));continue
            try:d=json.loads(summary.read_text())
            except (OSError,ValueError) as e:
                rows.append(dict(arm=arm,status='missing',error=str(e)));continue
            if d['status']!='pass':
                rows.append(dict(arm=arm,status='fail',stage=d.get('stage'),error=d.get('traceback')));continue
            latent=torch.load(case/'latents.pt',map_location='cpu',weights_only=True)
            if arm=='none':base=latent
            if arm=='global':global_latent=latent
            prefix_exact=None if arm=='window128' or base is None else torch.equal(latent[:,:96],base[:,:96])
            before_expiry=None if arm!='global_one_chunk' or global_latent is None else torch.equal(latent[:,:104],global_latent[:,:104])
            frames=[];native_keyframes={}
            with av.open(str(case/'video.mp4')) as container:
                for index,frame in enumerate(container.decode(video=0)):
                    frames.append(frame.reformat(width=208,height=120).to_ndarray(format='rgb24'))
                    if index in native_review_indices()[0]:
                        dest=args.output/f'lane{lane}__{arm}__native_frame{index}.png'
                        frame.to_image().save(dest);native_keyframes[f'native_frame{index}']=dest.name
            if len(frames)!=509:raise ValueError('truncated full video')
            boards=dict(native_keyframes)
            periods={'source':(40,48),'away':(64,96),'first_return':(96,104),'late_return':(104,128),
                     'quarter1':(0,32),'quarter2':(32,64),'quarter3':(64,96),'quarter4':(96,128)}
            for name,(a,b) in periods.items():
                dest=args.output/f'lane{lane}__{arm}__{name}.png'
                storyboard(frames,np.linspace(pixel_start(a),pixel_start(b)-1,16).round().astype(int),dest)
                boards[name]=dest.name
            # Exhaustive final-away board establishes actual absence; all 128 frames.
            for page in range(4):
                dest=args.output/f'lane{lane}__{arm}__away_all_{page}.png'
                storyboard(frames,np.arange(253+32*page,285+32*page),dest)
                boards[f'away_all_{page}']=dest.name
            # Compact comparison, denser boards, and native-resolution keyframes.
            indices=native_review_indices()[1]
            panel=Image.new('RGB',(8*224,158),'white');draw=ImageDraw.Draw(panel)
            draw.text((4,2),f'lane{lane} {arm}: SOURCE | AWAY | FIRST RETURN | LATE RETURN',fill='black')
            for col,index in enumerate(indices):
                thumb=Image.fromarray(frames[index]).resize((224,123))
                panel.paste(thumb,(col*224,30));draw.text((col*224+3,17),str(index),fill='black')
            panels.append(panel)
            with (case/'video.mp4').open('rb') as video_handle:
                video_sha=hashlib.file_digest(video_handle,'sha256').hexdigest()
            row=dict(arm=arm,status='technical_pass',prefix_exact=prefix_exact,before_expiry_exact=before_expiry,
                boards=boards,summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),
                video_sha256=video_sha,
                latent_sha256=d['latent_sha256'],gpu=d['gpu'],local_frames=d['local_frames'],
                generation_s=d['native_DiT_s'],VAE_s=d['native_VAE_s'],load_s=d['load_s'],
                wall_including_loading_s=d['wall_including_loading_s'],
                generation_peak_allocated_bytes=d['generation_peak_allocated_bytes'],
                native_KV_bytes=d['native_positive_and_negative_KV_bytes'],
                ledger=d.get('episode_memory',{}).get('ledger'),
                semantic_status='pending_assistant_visual_review_not_blind_human_score')
            rows.append(row);del frames
        if panels:
            combined=Image.new('RGB',(panels[0].width,sum(p.height for p in panels)),'white')
            offset=0
            for panel in panels:combined.paste(panel,(0,offset));offset+=panel.height
            combined.save(args.output/f'lane{lane}__comparison.jpg',quality=95)
        errors=[r['arm'] for r in rows if r.get('prefix_exact') is False or r.get('before_expiry_exact') is False]
        groups.append(dict(lane=lane,cases=rows,pairing_errors=errors))
        parts=[f'<h2>组 {lane}</h2>']
        if panels:parts.append(f'<img src="lane{lane}__comparison.jpg" width="100%">')
        for row in rows:
            links=' · '.join(f'<a href="{html.escape(path)}">{html.escape(name)}</a>' for name,path in row.get('boards',{}).items())
            parts.append(f'<p>{html.escape(row["arm"])} ({html.escape(row["status"])}): {links}</p>')
        sections.append(''.join(parts))
    report=dict(terminal=terminal,groups=groups,context_study=args.context_study,baseline_only=args.baseline_only,available_only=args.available_only,blind_human_review=False,semantic_review_complete=False)
    (args.output/'review_evidence.json').write_text(json.dumps(report,indent=2)+'\n')
    document='<!doctype html><html lang="zh"><meta charset="utf-8"><title>原生记忆使用对照</title><style>body{font:16px system-ui;max-width:1800px;margin:24px auto;padding:0 16px}img{border:1px solid #ddd}a{color:#075aa8}</style><h1>记忆使用：放置、寿命与完整窗口</h1><p>每组以自身source评估。技术通过不代表语义成功；大窗口允许不同早期轨迹。以下包括全部视频季度板和离开末段128帧。</p>'+''.join(sections)+'</html>'
    (args.output/'index.html').write_text(document)
    print(json.dumps(dict(groups=len(groups),cases=sum(len(g['cases']) for g in groups),semantic_review_complete=False)))


if __name__=='__main__':main()
