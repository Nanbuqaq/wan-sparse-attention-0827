#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--control',type=Path,required=True)
    p.add_argument('--gate',action='store_true');p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2)
    reference=json.loads((args.control/'summary.json').read_text());base=torch.load(args.control/'latents.pt',map_location='cpu',weights_only=True)
    target=48 if args.gate else 96;tokens=128 if args.gate else 880;rows=[]
    for condition in ('past','current'):
        root=args.root/condition
        try:
            d=json.loads((root/'summary.json').read_text());m=d['episode_memory'];ledger=m['ledger'];install=m['installation']
            assert d['status']=='pass' and d['memory_reconstruction']==condition
            for key in ('seed','cut_scenario','noise_sha256','pre_return_latent_sha256','latent_shape','fixed_native_adaln_recipe','source_files_sha256'):
                assert d[key]==reference[key],key
            assert torch.equal(torch.load(root/'latents.pt',map_location='cpu',weights_only=True)[:,:target],base[:,:target])
            assert install['all_GPU_KV_storage_pointers_preserved'] and install['not_exact_original_KV_replay']
            assert ledger['reconstruction_calls']==1 and ledger['CPU_archive_peak_bytes']<=16*1024**2
            assert ledger['source_replication_D2D_bytes']==8*tokens*24*128*2*2*30
            after=install['cache_metadata_transition']['after']
            assert after==dict(global_end_index=target*tokens,local_end_index=16*tokens,pinned_start=8*tokens,pinned_len=8*tokens)
            expected=[dict(query_start_latent=f,Q=8*tokens,K=(24 if i==0 else 32)*tokens,calls=150)
                      for i,f in enumerate(range(target,d['latent_frames'],8))]
            assert m['observed_return_attention_shapes']==expected
            assert m['source_pin_override']['KV_data_copied_bytes']==0
            assert ledger['demand_H2D_payload_bytes']==ledger['reconstruction_H2D_bytes']
            rows.append(dict(condition=condition,status='pass',ledger=ledger,pre_return_exact=True,
                new_KV_version_sha256=install['KV_storage_version_sha256'],capture=m['capture'],
                original_KV_equivalence_claimed=False,summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest()))
        except (OSError,ValueError,KeyError,AssertionError) as error:rows.append(dict(condition=condition,status='fail',error=str(error)))
    if all(r['status']=='pass' for r in rows):
        assert rows[0]['capture']['latent_sha256']==rows[1]['capture']['latent_sha256']
        assert rows[0]['new_KV_version_sha256']!=rows[1]['new_KV_version_sha256']
    result=dict(status='pass' if all(r['status']=='pass' for r in rows) else 'fail',cases=rows,expected=2,
        quality_not_inferred=True,gate=args.gate)
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
    if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
