#!/usr/bin/env python3
"""Collect causal full regressions; output equivalence is not a new quality sample."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback


REFERENCES={
    'bead_fill_s13':'key_position509_v1/related_recent',
    'settled_s19':'settled_memory509_v1/seed20260919/recent_virtual',
    'settled_s20':'settled_memory509_v1/seed20260920/recent_virtual',
    'toy_s13':'key_position_toy13_v1/related_recent',
    'toy_s21':'toy_position_seed21_v1/related_recent',
}


def main():
    import av
    import torch
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--references-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--available-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);rows=[]
    for name,relative in REFERENCES.items():
        root=args.root/name;path=root/'summary.json';refroot=args.references_root/relative
        if not path.exists():rows.append(dict(case=name,status='pending' if args.available_only else 'missing'));continue
        try:
            d=json.loads(path.read_text());ref=json.loads((refroot/'summary.json').read_text())
            assert d['status']==ref['status']=='pass' and d['observer_noise_latent_RGB_equivalence']
            for key in ('seed','noise_sha256','latent_sha256','latent_shape','prompts_per_block','native_shot_pin_events',
                        'source_files_sha256','assets_manifest_sha256','triton_version','fixed_native_adaln_recipe'):
                assert d[key]==ref[key],key
            assert d['latent_shape']==[1,128,48,44,80]
            m=d['causal_scene_memory'];assert len(m['archives'])==3 and len(m['installations'])==1
            initial=16 if name.startswith('settled') else 24
            assert [r['source_end'] for r in m['archives']]==[initial,48,96]
            install=m['installations'][0];plan=install['installation']['admission_plan']
            assert install['at_latent']==96 and plan['source_frames']==list(range(40,48)) and plan['temporal_delta']==64
            assert m['ledger']['history_H2D_KV_bytes']==2595225600 and m['ledger']['archive_D2H_KV_bytes']==3*2595225600
            assert m['ledger']['CPU_archive_peak_tensor_bytes']<=m['archive_budget_bytes']
            assert torch.equal(torch.load(root/'latents.pt',map_location='cpu',weights_only=True),
                               torch.load(refroot/'latents.pt',map_location='cpu',weights_only=True))
            hashes=[]
            for video_root in (root,refroot):
                digest=hashlib.sha256();count=0
                with av.open(str(video_root/'video.mp4')) as container:
                    for frame in container.decode(video=0):digest.update(frame.to_ndarray(format='rgb24').tobytes());count+=1
                assert count==509;hashes.append(digest.hexdigest())
            assert hashes[0]==hashes[1]
            rows.append(dict(case=name,status='pass',seed=d['seed'],full_actual_latent_and_decoded_RGB_exact=True,
                decoded_RGB_sha256=hashes[0],source_frames=plan['source_frames'],ledger=m['ledger'],
                native_DiT_s=d['native_DiT_s'],reference_DiT_s=ref['native_DiT_s'],
                single_run_timing_not_a_speedup_gate=True,quality_inherited_from_reference_not_new_sample=True,
                reference=str(refroot),case_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        except Exception:
            rows.append(dict(case=name,status='audit_fail',traceback=traceback.format_exc()))
    counts={state:sum(r['status']==state for r in rows) for state in ('pass','audit_fail','pending','missing')}
    report=dict(status='pass' if counts['pass']==5 else 'partial' if args.available_only and not counts['audit_fail'] else 'fail',
        cases=rows,counts=counts,expected_video_regressions=5,independent_new_quality_samples=0,
        scene_selection_scope='structured current-cue condition baseline, not general entity tracking',
        source_and_target_expectations_used_only_by_this_offline_audit=True)
    with args.output.open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')
    print(json.dumps(dict(status=report['status'],counts=counts)))


if __name__=='__main__':main()
