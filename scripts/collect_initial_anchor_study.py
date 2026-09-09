#!/usr/bin/env python3
"""Exact pre-return, explicit multiplicity and real transfer/execution contracts."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(2);control=json.loads((args.control/'summary.json').read_text())
    base=torch.load(args.control/'latents.pt',map_location='cpu',weights_only=True);rows=[]
    for policy in ('source_only','source_repeat'):
        for mode in ('raw_reveal','raw_away'):
            root=args.root/policy/mode
            try:
                d=json.loads((root/'summary.json').read_text());m=d['episode_memory'];plan=m['installation']['admission_plan']
                assert d['status']=='pass' and d['latent_shape']==[1,128,48,44,80] and d['pixels']['frames']==509
                assert d['initial_anchor_policy']==policy and d['local_frames']==32 and d['triton_version']=='3.2.0'
                assert d['native_KV_allocation_policy']=='CFG1_positive_only' and d['native_positive_and_negative_KV_bytes']==10380902400
                for key in ('seed','cut_scenario','noise_sha256','pre_return_latent_sha256','assets_manifest_sha256','source_files_sha256'):
                    assert d[key]==control[key],key
                assert torch.equal(torch.load(root/'latents.pt',map_location='cpu',weights_only=True)[:,:96],base[:,:96])
                frames=list(range(40,48)) if mode=='raw_reveal' else list(range(56,64))
                assert m['capture']['source_frames']==frames and plan['source_frames']==frames
                mult=1 if policy=='source_only' else 2
                assert plan['source_multiplicity']==mult and plan['visible_history_source_frames']==frames*mult
                assert plan['destination_token_range']==[0,7040*mult]
                assert m['ledger']['demand_H2D_payload_bytes']==m['ledger']['CPU_archive_peak_bytes']==2595225600
                assert m['ledger']['source_replication_D2D_bytes']==(0 if mult==1 else 2595225600)
                after=m['installation']['cache_metadata_transition']['after']
                assert after==dict(global_end_index=84480,local_end_index=7040*mult,
                                   pinned_start=-1 if mult==1 else 7040,pinned_len=0 if mult==1 else 7040)
                expected=[dict(query_start_latent=f,Q=7040,K=k*880,calls=150) for f,k in zip((96,104,112,120),
                          (16,24,32,32) if mult==1 else (24,32,32,32))]
                assert m['observed_return_attention_shapes']==expected
                row=dict(policy=policy,mode=mode,status='pass',actual_pre96_latent_exact=True,
                         source_summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),
                         admission_sha256=m['installation']['admission_plan_sha256'],ledger=m['ledger'])
            except (OSError,ValueError,KeyError,AssertionError) as error:
                row=dict(policy=policy,mode=mode,status='fail',error=str(error))
            rows.append(row)
    result=dict(status='pass' if all(r['status']=='pass' for r in rows) else 'fail',cases=rows,expected_cases=4,
                quality_not_inferred=True,missing=sum(not (args.root/r['policy']/r['mode']/'summary.json').exists() for r in rows))
    with (args.root/'initial_anchor_terminal.json').open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
