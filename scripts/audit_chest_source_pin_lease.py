#!/usr/bin/env python3
"""Delayed-prefix and actual resident-sample audit for source-pin lifetime."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.review_chest_causal_memory import audit_memory


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
        for key in ('noise_sha256','pre_return_latent_sha256','prompts_per_block','source_files_sha256','assets_manifest_sha256',
                    'fixed_native_adaln_recipe','triton_version','native_positive_and_negative_KV_bytes'):
            assert d[key]==ref[key],key
        assert d['constructor_mode']==ref['constructor_mode']=='reference'
        registration=ROOT/'configs/system/native_chest_source_pin_lease.json'
        assert d['object_state_protocol']['source_pin_lease_registration_sha256']==hashlib.sha256(registration.read_bytes()).hexdigest()
        m=d['causal_scene_memory'];details=audit_memory(m,'recent_virtual')
        assert m['archives']==ref['causal_scene_memory']['archives'] and m['decisions']==ref['causal_scene_memory']['decisions']
        for key in ('history_H2D_KV_bytes','archive_D2H_KV_bytes','CPU_archive_peak_tensor_bytes'):
            assert m['ledger'][key]==ref['causal_scene_memory']['ledger'][key],key
        lease=d['source_pin_lease'];assert len(lease['events'])==1
        event=lease['events'][0]
        assert event['completed_latent']==104 and event['source_frames']==list(range(40,48))
        assert event['requested_native_pin']==[21120,7040] and event['effective_source_pin']==[7040,7040]
        assert d['native_shot_pin_events'][:2]==ref['native_shot_pin_events'][:2]
        assert d['native_shot_pin_events'][2]==dict(completed_latent=104,pinned_start=7040,pinned_tokens=7040)
        assert [r['query_start_latent'] for r in lease['checks']]==[104,112,120]
        assert all(r['sampled_resident_KV_exact'] for r in lease['checks'])
        assert lease['ledger']['extra_raw_KV_H2D_bytes']==0 and lease['ledger']['metadata_GPU_write_bytes']==480
        assert lease['ledger']['witness_index_H2D_bytes']==256 and lease['ledger']['witness_KV_D2H_bytes']==4718592
        actual=torch.load(args.case/'latents.pt',map_location='cpu',weights_only=True)
        original=torch.load(args.reference/'latents.pt',map_location='cpu',weights_only=True)
        assert torch.isfinite(actual).all() and list(actual.shape)==list(original.shape)==d['latent_shape']
        assert torch.equal(actual[:,:104],original[:,:104])
        hashes=[];panels=[];boards={}
        for label,root in [('natural_repin',args.reference),('source_lease',args.case)]:
            frames=[];digest=hashlib.sha256()
            with av.open(str(root/'video.mp4')) as video:
                video.streams.video[0].codec_context.thread_count=2
                for i,frame in enumerate(video.decode(video=0)):
                    if i<413:digest.update(frame.to_ndarray(format='rgb24').tobytes())
                    frames.append(frame.reformat(width=224,height=123).to_ndarray(format='rgb24'))
                    if i in (188,412,444,476,508):frame.to_image().save(args.output/f'{label}__native{i}.png')
            assert len(frames)==509;hashes.append(digest.hexdigest())
            for name,(a,b) in dict(source=(157,189),first_return=(381,413),second_return=(413,445),
                    third_return=(445,477),last_return=(477,509),quarter1=(0,125),quarter2=(125,253),quarter3=(253,381),quarter4=(381,509)).items():
                target=args.output/f'{label}__{name}.png';storyboard(frames,np.linspace(a,b-1,16).round().astype(int),target)
                boards[label+'__'+name]=target.name
            panel=Image.new('RGB',(1792,150),'white');draw=ImageDraw.Draw(panel);draw.text((4,2),f'{args.seed} {label}',fill='black')
            for col,i in enumerate((188,380,412,420,444,460,484,508)):
                panel.paste(Image.fromarray(frames[i]),(224*col,25));draw.text((224*col+3,12),str(i),fill='black')
            panels.append(panel)
        assert hashes[0]==hashes[1]
        comparison=Image.new('RGB',(1792,300),'white')
        for i,panel in enumerate(panels):comparison.paste(panel,(0,150*i))
        comparison.save(args.output/'comparison.jpg',quality=95)
        result.update(status='pass',seed=args.seed,actual_pre104_and_pre413_decoded_RGB_exact=True,
            same_source_admission_and_raw_KV_transfer=True,actual_effective_pin_records_verified=True,
            sampled_source_KV_remains_exact_at_later_chunks=True,full_KV_identity_not_claimed=True,
            details=details,lease=lease,boards=boards,changes_which_other_context_is_evicted=True,
            summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference_summary_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest())
        (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Source-pin lease diagnostic</title>'
            '<style>body{font:17px system-ui;max-width:1800px;margin:25px auto}img{width:100%}</style>'
            '<h1>Same hybrid: natural repin versus original-source lease</h1><p>Actual pre104/pre413 identical; delayed route change, not layout equality.</p>'
            '<img src="comparison.jpg">'+''.join(f'<p><a href="{value}">{name}</a></p>' for name,value in boards.items()))
    except Exception:result.update(status='fail',traceback=traceback.format_exc());raise
    finally:
        (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
