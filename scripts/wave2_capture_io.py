"""Read-only fixed input adapters for new steady and existing full-source captures."""
import json
import torch


def load_call(case, *, legacy_source=False):
    summary=json.loads((case/'summary.json').read_text())
    if not legacy_source:
        return torch.load(case/'wave2_diagnostics.pt',weights_only=True,map_location='cpu')['call'],summary
    payload=torch.load(case/'attention_teacher.pt',weights_only=True,map_location='cpu',mmap=True)
    records=[r for r in payload['records'] if r['layer']==14 and r['phase']==0]
    if len(records)!=1 or records[0]['query_mode']!='full':
        raise ValueError('expected preregistered layer14/phase0 full query capture')
    row=records[0]
    routes=torch.load(case/'causal_block_routes.pt',weights_only=True,map_location='cpu')['records']
    matches=[r for r in routes if r['layer']==14 and r['phase']==0 and r['frame']==row['query_frame']]
    if len(matches)!=1 or not matches[0]['full_source_canonical']:
        raise ValueError('expected witnessed original full-source route')
    route=matches[0];start=route['destination_start']
    actual=[r for r in summary['causal_block_memory']['rows'] if r['layer']==14
            and r['phase']==0 and r['frame']==row['query_frame'] and r['active_source']]
    if len(actual)!=1:raise ValueError('missing independently logged source length')
    length=actual[0]['selected_source_tokens_per_head']
    if route.get('visible_source_tokens',length)!=length:raise ValueError('source length witnesses differ')
    end=start+length
    ft=row['frame_tokens'];blocks=[]
    for frame in range(start,end,ft):
        blocks.extend([list(range(frame+a,frame+min(a+64,ft))) for a in range(0,ft,64)])
    k,v=row['k'],row['v']
    km=torch.stack([k[0,idx].float().mean(0) for idx in blocks])
    vm=torch.stack([v[0,idx].float().mean(0) for idx in blocks])
    result=dict(q=row['q'],k=k,v=v,key_mean=km,value_mean=vm,counts=torch.tensor([len(x) for x in blocks]),
        token_blocks=blocks,protected=list(range(start))+list(range(end,k.shape[1])),frame_tokens=ft,
        input_scope='existing full-source recall: optional domain is actual installed source only; not a new steady video',
        source_witness=dict(frame=row['query_frame'],layer=14,phase=0,destination=[start,end],archive_version=route['archive_version']))
    return result,summary
