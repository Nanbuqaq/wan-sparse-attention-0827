#!/usr/bin/env python3
"""A new same-compiler property; never relabel the old cross-compiler failure."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--positive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(2)
    a=json.loads((args.native/'summary.json').read_text());b=json.loads((args.positive/'summary.json').read_text())
    assert a['status']=='pass'
    assert b['status']=='fail' and 'observer changed generated trajectory' in b['traceback']
    for d in (a,b):
        assert d['latent_shape']==[1,64,48,16,32] and d['local_frames']==128
        assert d['triton_version']=='3.3.1' and d['torch_dynamo_disabled'] and d['pixels']['frames']==253
        assert d['attention_backend']=='native_FA2' and not d['fallback_allowed']
    for field in ('noise_sha256','latent_sha256','seed','cut_scenario','gpu','assets_manifest_sha256','source_files_sha256','fixed_native_adaln_recipe'):
        assert a[field]==b[field],field
    assert a['pixels']['raw_RGB_sha256']==b['pixels']['raw_RGB_sha256']
    assert a['native_positive_and_negative_KV_bytes']==2*b['native_positive_and_negative_KV_bytes']
    assert torch.equal(torch.load(args.native/'latents.pt',map_location='cpu',weights_only=True),torch.load(args.positive/'latents.pt',map_location='cpu',weights_only=True))
    result=dict(status='pass',property='same_compiler_native_vs_positive_allocation_artifact_equivalence',
        native_case_status=a['status'],positive_case_original_status=b['status'],
        original_cross_compiler_failure_preserved=True,cross_compiler_equivalence_claimed=False,
        actual_full_latents_and_preencoded_RGB_exact=True,KV_bytes_halved=True,
        sources={str(path):hashlib.sha256((path/'summary.json').read_bytes()).hexdigest() for path in (args.native,args.positive)})
    with args.output.open('x') as handle:json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
