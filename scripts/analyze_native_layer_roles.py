#!/usr/bin/env python3
"""Full-output gate plus all-layer sampled role diagnostics; never online scores."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_native_attention_teacher import output_error,summary


def main():
    import av
    import torch
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);report=dict(status='running',rows=[])
    try:
        path=args.case/'summary.json';refpath=args.reference/'summary.json'
        d=json.loads(path.read_text());ref=json.loads(refpath.read_text())
        assert d['status']==ref['status']=='pass' and d['observer_noise_latent_RGB_equivalence']
        for key in ('seed','latent_shape','noise_sha256','latent_sha256','prompts_per_block','native_shot_pin_events',
                    'source_files_sha256','assets_manifest_sha256','triton_version','fixed_native_adaln_recipe'):
            assert d[key]==ref[key],key
        assert d['latent_shape']==[1,128,48,44,80]
        assert torch.equal(torch.load(args.case/'latents.pt',map_location='cpu',weights_only=True),
                           torch.load(args.reference/'latents.pt',map_location='cpu',weights_only=True))
        hashes=[]
        for root in (args.case,args.reference):
            digest=hashlib.sha256();count=0
            with av.open(str(root/'video.mp4')) as video:
                video.streams.video[0].codec_context.thread_count=2
                for frame in video.decode(video=0):digest.update(frame.to_ndarray(format='rgb24').tobytes());count+=1
            assert count==509;hashes.append(digest.hexdigest())
        assert hashes[0]==hashes[1]
        payload=args.case/'layer_role_probe.pt'
        with payload.open('rb') as h:payload_sha=hashlib.file_digest(h,'sha256').hexdigest()
        assert payload_sha==d['layer_role_probe']['sha256']
        data=torch.load(payload,map_location='cpu',weights_only=True)
        assert data['schema']=='native_layer_role_probe_v1' and data['complete'] and data['offline_teacher_only']
        assert data['full_QKV_not_stored'] and len(data['records'])==180
        expected={(f,p,l) for f in (96,120) for p in (0,3,4) for l in range(30)}
        assert {(r['query_frame'],r['phase'],r['layer']) for r in data['records']}==expected
        for record in data['records']:
            z=record['log_z'];values=record['role_outputs'];mass=z.softmax(-1)
            full=(mass[...,None]*values).sum(-2)
            error=output_error(full,record['native_output']);gate=error['max_abs']<=.02 and error['relative_l2']<=.01 and error['one_minus_cosine']<=.001
            denom=full.norm(dim=-1).clamp_min(1e-12);roles={}
            for index,name in enumerate(record['labels']):
                other=[j for j in range(4) if j!=index]
                dropped=(z[...,other].softmax(-1)[...,None]*values[...,other,:]).sum(-2)
                roles[name]=dict(probability_mass=summary(mass[...,index]),
                    removal_output_change=summary((dropped-full).norm(dim=-1)/denom))
            report['rows'].append(dict(query_frame=record['query_frame'],phase=record['phase'],layer=record['layer'],
                FP32_replay_gate=gate,FP32_replay_error=error,roles=roles))
        failed=sum(not r['FP32_replay_gate'] for r in report['rows'])
        report.update(status='pass' if failed==0 else 'numerical_gate_fail',actual_full_latent_and_decoded_RGB_exact=True,
            records=180,numerical_gate_failures=failed,seed=d['seed'],payload_sha256=payload_sha,
            summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),
            offline_only=True,no_layer_allowlist_selected=True,geometric_query_sample_not_full_Q=True,
            roles_at_96=['initial','source','away','current'],roles_at_120=['initial','first_return_anchor','recent_return','current'],
            late_roles_are_derived_from_verified_native_pin_lineage=True,clean_commit_not_extra_denoising_step=True,
            timing_not_comparable_to_unobserved_case=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        fig,axes=plt.subplots(2,3,figsize=(15,7),sharex=True,constrained_layout=True)
        for ri,frame in enumerate((96,120)):
            for ci,phase in enumerate((0,3,4)):
                selected=sorted((r for r in report['rows'] if r['query_frame']==frame and r['phase']==phase),key=lambda r:r['layer'])
                labels=list(selected[0]['roles']);matrix=np.array([[r['roles'][name]['probability_mass']['mean'] for r in selected] for name in labels])
                for col,r in enumerate(selected):
                    if not r['FP32_replay_gate']:matrix[:,col]=np.nan
                ax=axes[ri,ci];im=ax.imshow(matrix,vmin=0,vmax=1,aspect='auto',cmap='viridis')
                ax.set_title(f'query {frame} | '+{0:'first denoise',3:'last denoise',4:'clean commit'}[phase])
                ax.set_yticks(range(4),labels);ax.set_xticks([0,4,9,14,19,24,29]);ax.set_xlabel('DiT layer')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Sampled attention mass',shrink=.8)
        fig.suptitle(f'Offline all-layer role profile | seed {d["seed"]}; failed numeric rows masked')
        fig.savefig(args.output/'role_mass.png',dpi=130);plt.close(fig)
        (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>All-layer role probe</title>'
            '<style>body{font:18px system-ui;max-width:1600px;margin:30px auto}img{width:100%}</style>'
            '<h1>All-layer sampled role profile</h1><p>Actual complete outputs equal the saved hybrid. Not full-Q attention, semantic quality, online routing or a frozen layer policy. Clean commit is separate.</p>'
            '<img src="role_mass.png"><p><a href="analysis.json">All numeric gates, group deletion diagnostics and source hashes</a></p>')
    except Exception:report.update(status='fail',traceback=traceback.format_exc());raise
    finally:
        (args.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
