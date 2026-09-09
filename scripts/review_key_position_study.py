#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def selected_source_pixel_interval(source_start, frames=8):
    """Noninitial latent block uses [4*start-3, 4*end-3) pixel convention."""
    if source_start < 1 or frames < 1:
        raise ValueError('expected a noninitial committed source block')
    return 4*source_start-3, 4*(source_start+frames)-3


def main():
    import av
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from scripts.build_video_review_storyboards import storyboard
    from scripts.review_native_memory_study import native_review_indices
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--available-only',action='store_true')
    p.add_argument('--components',action='store_true');p.add_argument('--recent-control',type=Path);args=p.parse_args()
    if args.components and args.recent_control is None:p.error('--components requires --recent-control')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    if args.components:
        specs=[('original',args.control,40,'original'),('phase_only',args.root/'phase_only',40,'phase_only'),
               ('age_only',args.root/'age_only',40,'age_only'),('recent_virtual',args.recent_control,40,'recent_virtual')]
    else:
        specs=[('related_original',args.control,40,'original'),('related_recent',args.root/'related_recent',40,'recent_virtual'),
               ('away_original',args.root/'away_original',56,'original'),('away_recent',args.root/'away_recent',56,'recent_virtual')]
    original=json.loads((args.control/'summary.json').read_text());base=torch.load(args.control/'latents.pt',map_location='cpu',weights_only=True)
    rows=[];panels=[];prefix=None
    for name,root,source_start,policy in specs:
        path=root/'summary.json'
        if not path.exists() and args.available_only:rows.append(dict(case=name,status='pending'));continue
        d=json.loads(path.read_text())
        if d['status']!='pass':rows.append(dict(case=name,status='fail',error=d.get('traceback')));continue
        assert d['latent_shape']==[1,128,48,44,80] and d['local_frames']==32 and d['sink_frames']==8
        for key in ('seed','cut_scenario','noise_sha256','pre_return_latent_sha256','fixed_native_adaln_recipe','source_files_sha256',
                    'prompts_per_block','upstream_source_SHA','assets_manifest_sha256','attention_backend','KV_and_generator_dtype',
                    'triton_version','native_KV_allocation_policy','native_shot_pin_events'):
            assert d[key]==original[key],key
        latent=torch.load(root/'latents.pt',map_location='cpu',weights_only=True);assert torch.equal(latent[:,:96],base[:,:96])
        m=d['episode_memory'];plan=m['installation']['admission_plan']
        assert plan['source_frames']==list(range(source_start,source_start+8))
        assert m['ledger']['demand_H2D_payload_bytes']==m['ledger']['CPU_archive_peak_bytes']==2595225600
        assert m['ledger']['archive_D2H_payload_bytes']==2595225600
        assert m['ledger']['demand_D2D_KV_bytes']==0
        assert m['installation']['cache_metadata_unchanged'] and plan['destination_token_range']==[7040,14080]
        if policy!='original':
            assert d['episode_position_policy']==policy
            age=88-source_start if policy!='phase_only' else 0
            phase=(16 if source_start==40 else 8) if policy!='age_only' else 0
            assert plan['temporal_delta']==age+phase
            virtual_start=source_start+age
            assert plan['virtual_source_frames']==list(range(virtual_start,virtual_start+8))
            assert plan['spatial_key_channels_unchanged'] and plan['value_unchanged']
        frames=[];h=hashlib.sha256();boards={}
        with av.open(str(root/'video.mp4')) as container:
            for i,frame in enumerate(container.decode(video=0)):
                frames.append(frame.reformat(width=208,height=120).to_ndarray(format='rgb24'))
                if i<381:h.update(frame.to_ndarray(format='rgb24').tobytes())
                if i in native_review_indices()[0]:
                    target=args.output/f'{name}__native{i}.png';frame.to_image().save(target);boards[f'native{i}']=target.name
        assert len(frames)==509
        if prefix is None:prefix=h.hexdigest()
        assert prefix==h.hexdigest()
        for label,(a,b) in {'reference_state':(157,189),'selected_source':selected_source_pixel_interval(source_start),
                           'away':(253,381),'first_return':(381,413),'late_return':(413,509),
                           'quarter1':(0,125),'quarter2':(125,253),'quarter3':(253,381),'quarter4':(381,509)}.items():
            target=args.output/f'{name}__{label}.png';storyboard(frames,np.linspace(a,b-1,16).round().astype(int),target);boards[label]=target.name
        panel=Image.new('RGB',(1792,150),'white');draw=ImageDraw.Draw(panel);draw.text((4,2),name,fill='black')
        for col,i in enumerate(native_review_indices()[1]):
            panel.paste(Image.fromarray(frames[i]).resize((224,123)),(col*224,25));draw.text((col*224+3,12),str(i),fill='black')
        panels.append(panel);rows.append(dict(case=name,status='technical_pass',pre96_latent_and_decoded_prefix_exact=True,
            actual_decoded_prefix_sha256=h.hexdigest(),noise_sha256=d['noise_sha256'],seed=d['seed'],
            boards=boards,ledger=m['ledger'],admission_plan=plan,summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_summary=str(path),native_DiT_s=d['native_DiT_s'],native_VAE_s=d['native_VAE_s'],
            generation_peak_allocated_bytes=d['generation_peak_allocated_bytes']))
    canvas=Image.new('RGB',(1792,150*len(panels)),'white')
    for i,panel in enumerate(panels):canvas.paste(panel,(0,i*150))
    canvas.save(args.output/'comparison.jpg',quality=95)
    (args.output/'technical_audit.json').write_text(json.dumps(dict(cases=rows,available_snapshot=args.available_only,
        semantic_review_complete=False,source_V_and_spatial_K_preserved_by_operator_gate=True),indent=2)+'\n')
    title='历史K时间距离 × 镜头偏移' if args.components else '源内容 × 历史K位置'
    body=['<!doctype html><meta charset="utf-8"><title>历史K位置绑定</title><style>body{font:17px system-ui;max-width:1800px;margin:24px auto;padding:0 20px}img{width:100%}</style><h1>'+title+'</h1><p>同noise/pre96 latent及前381帧decodedRGB、同H2D、同逻辑容量；位置干预只改历史K时间旋转，不是layout优化。</p><img src="comparison.jpg">']
    for r in rows:body.append('<p>'+r['case']+' ('+r['status']+'): '+' · '.join(f'<a href="{v}">{k}</a>' for k,v in r.get('boards',{}).items())+'</p>')
    (args.output/'index.html').write_text(''.join(body));print(json.dumps(dict(cases=len(rows),available_snapshot=args.available_only)))


if __name__=='__main__':main()
