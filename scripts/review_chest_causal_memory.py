#!/usr/bin/env python3
"""Actual-prefix, causal admission and equal-byte audit; no automatic quality score."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def source_pixel_interval(frames):
    if not frames or frames!=list(range(frames[0],frames[-1]+1)):
        raise ValueError('contiguous actual source frames required')
    return max(0,4*frames[0]-3),4*(frames[-1]+1)-3


def audit_memory(memory,policy):
    assert memory['position_policy']==policy
    assert memory['selection_policy']==dict(minimum_cosine=.8,margin=.05,min_gap=32)
    assert memory['archive_budget_bytes']==8589934592
    assert not memory['raw_source_frames_or_target_frames_supplied_to_selector']
    assert not memory['future_text_or_generated_outputs_read_by_selector']
    assert memory['native_active_KV_capacity_unchanged']
    ledger=memory['ledger'];archives=memory['archives'];installs=memory['installations']
    assert ledger['CPU_archive_peak_tensor_bytes']<=memory['archive_budget_bytes']
    assert ledger['archive_D2H_KV_bytes']==sum(r['KV_bytes'] for r in archives)
    assert ledger['history_H2D_KV_bytes']==sum(r['transfer_ledger']['demand_H2D_payload_bytes'] for r in installs)
    selections=[]
    for entry in installs:
        plan=entry['installation']['admission_plan']
        source=next(r for r in archives if r['archive_version']==plan['archive_version'])
        assert plan['source_frames']==source['source_frames']
        assert plan['target_start']==entry['at_latent'] and source['source_end']<=entry['at_latent']-32
        assert entry['transfer_ledger']['demand_H2D_payload_bytes']==source['KV_bytes']
        assert entry['installation']['cache_metadata_unchanged']
        assert entry['installation']['native_attention_size_unchanged']
        if policy=='original':assert plan['temporal_delta']==0
        else:
            assert plan['temporal_delta']==entry['at_latent']-source['source_end']+entry['current_phase']-source['source_phase']
            assert plan['value_unchanged'] and plan['spatial_key_channels_unchanged']
        selections.append(dict(at_latent=entry['at_latent'],source_frames=plan['source_frames'],
            archive_version=plan['archive_version'],temporal_delta=plan['temporal_delta']))
    return dict(ledger=ledger,selections=selections,
        selected_expected_settled_source=any(r['at_latent']==96 and r['source_frames']==list(range(40,48)) for r in selections),
        offline_source_expectation_not_a_technical_pass_condition=True,abstained=not bool(installs))


def main():
    import av
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from scripts.build_video_review_storyboards import storyboard
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--control-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    rows=[];pairs=[];body=[]
    for seed in (20260925,20260926):
        control=args.control_root/f'seed{seed}'/'chest_revisit'
        base=json.loads((control/'summary.json').read_text())
        assert base['status']=='pass'
        reference=torch.load(control/'latents.pt',map_location='cpu',weights_only=True)
        prefix=None;memories={};panels=[]
        for policy,root in [('dense',control)]+[(k,args.root/f'seed{seed}'/k) for k in ('original','recent_virtual')]:
            row=dict(seed=seed,policy=policy,summary=str(root/'summary.json'))
            if not (root/'summary.json').exists():
                row['status']='missing';rows.append(row);continue
            try:
                path=root/'summary.json';d=json.loads(path.read_text())
                assert d['status']=='pass' and d['seed']==seed and d['pixel_frames']==509
                assert d['latent_shape']==[1,128,48,44,80]
                for key in ('noise_sha256','pre_return_latent_sha256','prompts_per_block','source_files_sha256',
                            'assets_manifest_sha256','fixed_native_adaln_recipe','triton_version','native_shot_pin_events'):
                    assert d[key]==base[key],key
                actual=torch.load(root/'latents.pt',map_location='cpu',weights_only=True)
                assert list(actual.shape)==d['latent_shape'] and torch.isfinite(actual).all()
                assert torch.equal(actual[:,:96],reference[:,:96]);del actual
                selections=[]
                if policy!='dense':
                    protocol=d['object_state_protocol']
                    assert not protocol['Dense_only'] and not protocol['formal_holdout']
                    registration=ROOT/'configs/system/native_object_state_memory.json'
                    assert protocol['memory_registration_sha256']==hashlib.sha256(registration.read_bytes()).hexdigest()
                    memory=d['causal_scene_memory'];details=audit_memory(memory,policy)
                    row.update(details);memories[policy]=memory;selections=details['selections']
                frames=[];digest=hashlib.sha256();boards={}
                with av.open(str(root/'video.mp4')) as video:
                    video.streams.video[0].codec_context.thread_count=2
                    for i,frame in enumerate(video.decode(video=0)):
                        frames.append(frame.reformat(width=224,height=123).to_ndarray(format='rgb24'))
                        if i<381:digest.update(frame.to_ndarray(format='rgb24').tobytes())
                        if i in (188,381,412,460,508):
                            target=args.output/f'seed{seed}__{policy}__native{i}.png'
                            frame.to_image().save(target);boards[f'native{i}']=target.name
                assert len(frames)==509
                if prefix is None:prefix=digest.hexdigest()
                assert prefix==digest.hexdigest()
                windows=dict(reference_source=(157,189),late_away=(253,381),first_return=(381,413),
                    late_return=(413,509),quarter1=(0,125),quarter2=(125,253),quarter3=(253,381),quarter4=(381,509))
                for j,selection in enumerate(selections):
                    windows[f'actual_selected_source{j}']=source_pixel_interval(selection['source_frames'])
                for label,(start,end) in windows.items():
                    target=args.output/f'seed{seed}__{policy}__{label}.png'
                    storyboard(frames,np.linspace(start,end-1,16).round().astype(int),target);boards[label]=target.name
                panel=Image.new('RGB',(1792,150),'white');draw=ImageDraw.Draw(panel)
                draw.text((4,2),f'{seed} {policy}',fill='black')
                for col,i in enumerate((156,188,313,380,385,412,460,508)):
                    panel.paste(Image.fromarray(frames[i]),(col*224,25));draw.text((col*224+3,12),str(i),fill='black')
                panels.append(panel)
                row.update(status='pass',actual_pre96_and_pre381_decoded_RGB_exact=True,
                    prefix_RGB_sha256=prefix,summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    boards=boards,native_DiT_s=d['native_DiT_s'],single_run_time_not_a_speedup_test=True)
            except Exception:row.update(status='fail',traceback=traceback.format_exc())
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('boards','traceback')}),flush=True)
        pair=dict(seed=seed,status='fail')
        if set(memories)=={'original','recent_virtual'}:
            a,b=memories['original'],memories['recent_virtual']
            same=(a['archives']==b['archives'] and a['decisions']==b['decisions'] and all(
                a['ledger'][k]==b['ledger'][k] for k in ('archive_D2H_KV_bytes','history_H2D_KV_bytes',
                    'condition_summary_D2H_bytes','CPU_archive_peak_tensor_bytes','evicted_archives')))
            pair.update(status='pass' if same else 'fail',same_archive_decisions_and_payload=same)
        pairs.append(pair)
        if panels:
            board=Image.new('RGB',(1792,150*len(panels)),'white')
            for i,panel in enumerate(panels):board.paste(panel,(0,150*i))
            board.save(args.output/f'seed{seed}__comparison.jpg',quality=95)
            body.append(f'<h2>Seed {seed}</h2><img src="seed{seed}__comparison.jpg">')
    counts={k:sum(r['status']==k for r in rows) for k in ('pass','fail','missing')}
    report=dict(status='pass' if counts['pass']==6 and all(p['status']=='pass' for p in pairs) else 'fail',
        rows=rows,pairs=pairs,counts=counts,new_video_executions=4,reused_dense=2,formal_holdout=False,
        semantic_review_complete=False,technical_pass_does_not_imply_state_recovery=True,
        visible_cut_control_also_loses_state=True)
    (args.output/'technical_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in rows:
        body.append('<p>'+html.escape(f"{row['seed']} {row['policy']} {row['status']}")+': '+
            ' | '.join(f'<a href="{value}">{key}</a>' for key,value in row.get('boards',{}).items())+'</p>')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Chest causal memory audit</title>'
        '<style>body{font:17px system-ui;max-width:1800px;margin:25px auto}img{width:100%}</style>'
        '<h1>Chest: same causal admission, original versus recent K positions</h1>'
        '<p>Technical/source/byte audit only. Review open lid, blue cloth, geometry, contamination and late action separately.</p>'+''.join(body))
    print(json.dumps(dict(status=report['status'],counts=counts,pairs=pairs)))


if __name__=='__main__':main()
