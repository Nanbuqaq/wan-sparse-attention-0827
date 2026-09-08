#!/usr/bin/env python3
"""Paired low-resolution destination/context/lifetime probe, not layout speedup."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = ('none', 'global', 'shot')


def validate_group(reports):
    if set(reports) != set(ARMS):
        raise ValueError('three destination arms required')
    for arm, report in reports.items():
        if (report['status'] != 'pass' or report['latent_shape'] != [1,64,48,16,32]
                or report['pixels']['frames'] != 253 or report['local_frames'] != 32
                or report['sink_frames'] != 8 or report['attention_backend'] != 'native_FA2'
                or report['fallback_allowed']):
            raise ValueError('incorrect technical gate, shape, capacity or backend')
        if report['episode_memory_mode'] != ('none' if arm == 'none' else 'raw_reveal'):
            raise ValueError('incorrect admission mode')
    fields = ('seed', 'cut_scenario', 'gpu', 'noise_sha256', 'pre_return_latent_sha256', 'runner_commit')
    if len({tuple(r[f] for f in fields) for r in reports.values()}) != 1:
        raise ValueError('unmatched case, recipe source, noise or pre-return prefix')
    if any(r['fixed_native_adaln_recipe'] != dict(num_warps=16, num_stages=1) for r in reports.values()):
        raise ValueError('unmatched numerical recipe')
    baseline = reports['none']['episode_memory']['ledger']
    if baseline['CPU_archive_peak_bytes'] or baseline['demand_H2D_payload_bytes']:
        raise ValueError('baseline retained hidden episode archive')
    plans = []
    for arm in ('global', 'shot'):
        memory = reports[arm]['episode_memory']
        install = memory['installation']
        plan = install['admission_plan'].copy()
        if (memory['destination_role'] != arm or install['at_latent'] != 48
                or memory['capture']['source_frames'] != list(range(8,16))
                or not install['cache_metadata_unchanged']
                or not memory['K_positions_preserved_not_rebased']):
            raise ValueError('wrong destination, causal source or position policy')
        a, b = plan.pop('destination_token_range')
        if b-a != 8*plan['frame_tokens'] or (a != 0 if arm == 'global' else a < 8*plan['frame_tokens']):
            raise ValueError('wrong or overlapping destination slot')
        plans.append(plan)
    if plans[0] != plans[1]:
        raise ValueError('source or admission setting changed beyond destination')
    memories = [reports[a]['episode_memory'] for a in ('global','shot')]
    for field in ('archive_D2H_payload_bytes', 'demand_H2D_payload_bytes', 'CPU_archive_peak_bytes'):
        if not memories[0]['ledger'][field] == memories[1]['ledger'][field] > 0:
            raise ValueError('unequal source byte budget')
    if memories[0]['installation']['admission_plan_sha256'] == memories[1]['installation']['admission_plan_sha256']:
        raise ValueError('changed destination must change admission SHA')
    return dict(status='pass', pre_return_and_noise_exact=True, same_source_coordinates_and_bytes=True,
                different_destination_SHA=True, attribution='placement_plus_displaced_context_plus_lifetime',
                not_pure_layout_optimization=True, quality='requires_visual_review')


def main():
    import numpy as np
    import torch
    from scripts.build_video_review_storyboards import decode_frames, storyboard
    from scripts.review_revisit_videos import pixel_start
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);groups=[]
    for lane in range(2):
        root=args.root/f'lane{lane}';reports={};errors=[];rows=[]
        for arm in ARMS:
            try: reports[arm]=json.loads((root/arm/'summary.json').read_text())
            except (OSError, ValueError) as error: errors.append(f'{arm}: {error}')
        try: contract=validate_group(reports)
        except (KeyError, ValueError) as error: contract=dict(status='fail',error=str(error))
        reference=None
        if reports.get('none',{}).get('status')=='pass':
            reference=torch.load(root/'none/latents.pt',map_location='cpu',weights_only=True)
        for arm in ARMS:
            d=reports.get(arm,{})
            if d.get('status')!='pass':
                rows.append(dict(arm=arm,status='fail',error=d.get('traceback','no terminal success')));continue
            latent=torch.load(root/arm/'latents.pt',map_location='cpu',weights_only=True)
            prefix=reference is not None and torch.equal(reference[:,:48],latent[:,:48])
            if not prefix: errors.append(f'{arm}: actual prefix mismatch')
            frames=decode_frames(root/arm/'video.mp4');boards={}
            if len(frames)!=253: errors.append(f'{arm}: truncated video')
            for part,(a,b) in {'source':(8,16),'away':(32,48),'first_return':(48,56),'late_return':(56,64)}.items():
                path=args.output/f'lane{lane}__{arm}__{part}.png'
                storyboard(frames,np.linspace(pixel_start(a),pixel_start(b)-1,16).round().astype(int),path)
                boards[part]=str(path)
            rows.append(dict(arm=arm,status='technical_pass',prefix_exact=prefix,boards=boards,
                changed_complete_latent_from_baseline=not torch.equal(latent,reference),
                ledger=d['episode_memory']['ledger'],installation=d['episode_memory']['installation'],
                shot_pin_events=d['native_shot_pin_events'],
                summary_sha256=hashlib.sha256((root/arm/'summary.json').read_bytes()).hexdigest()))
        groups.append(dict(lane=lane,contract=contract,cases=rows,errors=errors))
    report=dict(status='pass' if all(g['contract']['status']=='pass' and not g['errors'] for g in groups) else 'fail',
                groups=groups,cases=6,low_resolution_gate_not_full509_quality=True)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='groups'}))
    if report['status']!='pass': raise SystemExit(1)


if __name__=='__main__': main()
