#!/usr/bin/env python3
"""Deterministic source-only geometry proposal from small SAM region components."""
import argparse
import hashlib
import json
from pathlib import Path


def component_box(rows,width=1280,height=704,policy='largest'):
    if policy not in ('largest','all','mutual_geometry'):raise ValueError('explicit component policy required')
    nodes=[r for r in rows if 0<r['spatial_tokens']<=64]
    boxes={r['id']:(r['bbox'][0],r['bbox'][1],r['bbox'][0]+r['bbox'][2],r['bbox'][1]+r['bbox'][3]) for r in nodes}
    remaining=set(boxes);components=[]
    while remaining:
        todo=[min(remaining)];remaining.remove(todo[0]);component=[]
        while todo:
            i=todo.pop();component.append(i);a=boxes[i]
            for j in sorted(remaining.copy()):
                b=boxes[j];dx=max(0,max(a[0],b[0])-min(a[2],b[2]));dy=max(0,max(a[1],b[1])-min(a[3],b[3]))
                if dx*dx+dy*dy<=32*32:remaining.remove(j);todo.append(j)
        if len(component)>=3:components.append(sorted(component))
    if not components:raise ValueError('no registered multi-part source component')
    by_id={r['id']:r for r in nodes}
    components.sort(key=lambda c:(-len(c),-sum(by_id[i]['pixel_area'] for i in c),c[0]))
    if policy=='mutual_geometry' and len(components)>1:
        cb=[]
        for c in components:
            bs=[boxes[i] for i in c]
            cb.append((min(b[0] for b in bs),min(b[1] for b in bs),max(b[2] for b in bs),max(b[3] for b in bs)))
        nearest=[]
        for i,a in enumerate(cb):
            def distance(j):
                b=cb[j];dx=max(0,max(a[0],b[0])-min(a[2],b[2]));dy=max(0,max(a[1],b[1])-min(a[3],b[3]))
                return dx*dx+dy*dy,j
            nearest.append(min((j for j in range(len(cb)) if j!=i),key=distance))
        merged=[];used=set()
        for i,c in enumerate(components):
            if i in used:continue
            j=nearest[i]
            if nearest[j]==i:merged.append(sorted(c+components[j]));used.update((i,j))
            else:merged.append(c);used.add(i)
        components=sorted(merged,key=lambda c:(-len(c),-sum(by_id[i]['pixel_area'] for i in c),c[0]))
    selected=sorted(i for c in components for i in c) if policy=='all' else components[0]
    bounds=[boxes[i] for i in selected]
    box=[max(0,min(b[0] for b in bounds)-32),max(0,min(b[1] for b in bounds)-32),
         min(width,max(b[2] for b in bounds)+32),min(height,max(b[3] for b in bounds)+32)]
    return box,selected,components


def main():
    p=argparse.ArgumentParser();p.add_argument('--proposals',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--component-policy',choices=('largest','all','mutual_geometry'),default='largest');args=p.parse_args()
    d=json.loads(args.proposals.read_text())
    if d['manual_box_or_mask_input'] or d['return_or_future_pixel_input']:raise ValueError('source-only automatic producer required')
    box,selected,components=component_box(d['proposals'],policy=args.component_policy)
    report=dict(status='automatic_box_derived',bbox=box,selected_component_ids=selected,all_components=components,
        component_policy=args.component_policy,
        source_pixel_frame=d['source_pixel_frame'],source_pixel_sha256=d['source_pixel_sha256'],source_latent_sha256=d['source_latent_sha256'],
        pixel_input_kind=d.get('pixel_input_kind','decoded_video_rgb'),
        automatic_box=True,manual_mask_or_return_input=False,semantic_target_selection=False,
        proposal_report=str(args.proposals.resolve()),
        proposal_report_sha256=hashlib.sha256(args.proposals.read_bytes()).hexdigest(),
        rule='nodes<=64 spatial tokens; bbox gap<=one32-pixel token; >=3 nodes; mutual_geometry merges one pass of mutual bbox-nearest components; rank count/area/ID or keep all; pad one token',
        limitations=['largest part component is a geometry heuristic, not an identified semantic target','no current-query or future pixels used'])
    with args.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=='__main__':main()
