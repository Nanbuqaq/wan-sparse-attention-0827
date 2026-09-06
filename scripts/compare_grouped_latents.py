#!/usr/bin/env python3
"""Same-seed Dense latent comparisons before the separate video-quality pass."""
import argparse
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.offline_eval import output_error_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--states',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(2)
    cases=json.loads(Path(args.states).read_text())['cases'];groups={}
    for c in cases:groups.setdefault((c['prompt_id'],c['seed'],c['latent_frames']),[]).append(c)
    rows=[]
    for (prompt,seed,length),group in groups.items():
        dense=next(c for c in group if c['method']=='rag_dense')
        if not all(c['status']=='pass' for c in group):raise ValueError('failed case in latent group')
        ref=torch.load(Path(dense['video']).parent/'latents.pt',map_location='cpu',weights_only=True)
        for c in group:
            if c['method']=='rag_dense':continue
            if c['initial_noise_sha256']!=dense['initial_noise_sha256']:raise ValueError('noise SHA mismatch')
            cur=torch.load(Path(c['video']).parent/'latents.pt',map_location='cpu',weights_only=True)
            row={'id':c['id'],'prompt':prompt,'seed':seed,'latent_frames':length,'method':c['method'],
                 'relative_dense_latent_error':output_error_metrics(ref,cur),
                 'late_quarter_error':output_error_metrics(ref[:,3*length//4:],cur[:,3*length//4:])}
            rows.append(row);print(json.dumps(row),flush=True)
    result={'status':'pass','scope':'latent_fidelity_only_not_video_quality_promotion','rows':rows}
    with Path(args.output).open('x') as f:json.dump(result,f,indent=2)


if __name__=='__main__':main()
