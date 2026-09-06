#!/usr/bin/env python3
"""Compare optimized compilation to actual frozen routes on32 causal contexts."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adapters.longlive_sparse.archive import HistoryArchive
from adapters.longlive_sparse.config import SparseHistoryConfig
from adapters.longlive_sparse.selectors import PretransferQuerySummary
from adapters.longlive_sparse.route_plan import HistoryRoutePlan


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture-root',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(2)
    root=Path(args.capture_root)
    audit=json.loads((root/'trajectory_audit.json').read_text())
    if audit['status']!='pass':raise ValueError('original trajectories not verified')
    rows=[]
    for case in audit['cases']:
        if case['method']=='rag_dense':continue
        for path in sorted(Path(case['capture_dir']).glob('*pass00.pt')):
            cap=torch.load(path,map_location='cpu',weights_only=True)
            archive=HistoryArchive(SparseHistoryConfig.from_mapping(cap['sparse_config']),
                spatial_height=cap['spatial_height'],spatial_width=cap['spatial_width'])
            frames=list(dict.fromkeys(cap['frame_ids'][0,0].tolist()));context=cap['actual_online_context']
            for frame in frames:
                mask=cap['frame_ids'][0,0]==frame
                order=cap['token_ids'][0,0,mask].argsort()
                k=cap['key_unrotated'][:,mask][:,order];v=cap['value'][:,mask][:,order]
                index=archive.index_frame(0,frame,k,v)
                block=context['block_frame_ids']==frame
                archive._layers[0][frame]=replace(index,block_centroids=context['key_prototypes'][:,:,block],
                                                  block_value_centroids=context['value_prototypes'][:,:,block])
            plan=archive.route_indexed(0,PretransferQuerySummary(**cap['actual_query_summary']),frames,
                                      exact_k_tokens=cap['exact_key'].shape[1])
            old=HistoryRoutePlan.from_state_dict(cap['route_plan'])
            equal={key:(torch.equal(value,plan.state_dict()[key]) if isinstance(value,torch.Tensor) else value==plan.state_dict()[key])
                   for key,value in old.state_dict().items()}
            rows.append({'case':case['id'],'capture':path.name,'status':'pass' if all(equal.values()) else 'fail',
                         'all_fields_equal':equal,'same_route_sha':plan.digest()==old.digest()})
    result={'status':'pass' if len(rows)==32 and all(r['status']=='pass' for r in rows) else 'fail','cases':rows,
            'scope':'full_plan_including_metadata_exactly_preserved','expected':32}
    with Path(args.output).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':result['status'],'cases':len(rows)}))
    if result['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main()
