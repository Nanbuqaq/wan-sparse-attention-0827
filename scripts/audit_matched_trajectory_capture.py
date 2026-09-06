#!/usr/bin/env python3
"""Fail closed if instrumentation does not reproduce the specified video path."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.history_cache import tensor_sha256


def route_sequence(stats):
    return [(r['layer_id'],r['current_start'],r['denoising_pass'],r['route_plan_sha256'])
            for r in stats['call_records']]


def compare(reference, observed):
    return {'same_initial_noise':reference['initial_noise_sha256']==observed['initial_noise_sha256'],
            'same_case_identity':reference['case_key_sha256']==observed['case_key_sha256']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture-root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    root=Path(args.capture_root).resolve()
    protocol=json.loads((root/'control/protocol.json').read_text())
    original=json.loads((Path(protocol['reference_root'])/'states.json').read_text())['cases']
    observed=json.loads((root/'states.json').read_text())['cases']
    by_id={r['id']:r for r in observed}
    rows=[]
    for ref in original:
        cur=by_id.get(ref['id'])
        row={'id':ref['id'],'method':ref['method'],'prompt':ref['prompt_id']}
        if cur is None or cur['status']!='pass':
            rows.append({**row,'status':'fail','reason':'missing or failed diagnostic generator'})
            continue
        checks=compare(ref,cur)
        a=torch.load(Path(ref['video']).parent/'latents.pt',map_location='cpu',weights_only=True)
        b=torch.load(Path(cur['video']).parent/'latents.pt',map_location='cpu',weights_only=True)
        checks['bitwise_latents']=torch.equal(a,b)
        checks['ordered_routes']=route_sequence(json.loads(Path(ref['stats']).read_text()))==route_sequence(json.loads(Path(cur['stats']).read_text()))
        directory=Path(cur['video']).parent.parent/'complete_attention_captures'/cur['id']
        expected={f'layer{layer:02d}_start{start:08d}_pass{call:02d}.pt'
                  for layer in protocol['layers'] for start in protocol['starts'] for call in range(protocol['passes'])}
        found={p.name for p in directory.glob('*.pt')}
        checks['complete_capture_coverage']=found==expected
        rows.append({**row,'status':'pass' if all(checks.values()) else 'fail','checks':checks,
            'latent_sha256':tensor_sha256(b),'reference_latent_sha256':tensor_sha256(a),
            'capture_dir':str(directory),'captures':len(found)})
    result={'status':'pass' if len(rows)==protocol['reference_cases'] and all(r['status']=='pass' for r in rows) else 'fail',
        'scope':'exact_specified_39_latent_trajectory_not_a_120_latent_proxy',
        'generation_commit':protocol['generation_commit'],'cases':rows,
        'protocol_sha256':hashlib.sha256((root/'control/protocol.json').read_bytes()).hexdigest(),
        'new_quality_trial':False,'formal_promotion_allowed':False}
    out=Path(args.output)
    with out.open('x') as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))
    if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
